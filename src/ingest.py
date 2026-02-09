from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import unicodedata
import uuid
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import crawl_config as cfg


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_for_embedding(text: str) -> str:
    """Character-level sanitization to avoid embedding encoder JSON/NaN failures."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")

    cleaned_chars = []
    for ch in text:
        category = unicodedata.category(ch)
        # Remove non-printable/control/format/surrogate/private-use characters.
        if category.startswith("C") and ch not in ("\n", "\r", "\t"):
            continue
        cleaned_chars.append(ch)

    return "".join(cleaned_chars)


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    # Drop query string + fragment for dedup/stability.
    return urlunparse((scheme, netloc, path, "", "", ""))


def is_allowed_url(url: str) -> bool:
    lowered = url.lower()
    if any(deny in lowered for deny in cfg.DENY_URL_SUBSTRINGS):
        return False

    parsed = urlparse(url)
    if parsed.netloc.lower() not in cfg.ALLOWED_DOMAINS:
        return False

    return any(parsed.path.startswith(prefix) for prefix in cfg.ALLOWED_PATH_PREFIXES)


def classify_page_type(url: str) -> str:
    lowered = url.lower()
    if any(hint in lowered for hint in cfg.MUTABLE_PAGE_HINTS):
        return "mutable_live"
    return "news"


def extract_page_fields(html: str) -> tuple[str, str, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())

    published_at = None
    published_meta = soup.find("meta", attrs={"property": "article:published_time"})
    if published_meta and published_meta.get("content"):
        published_at = published_meta.get("content")
    elif soup.find("time") and soup.find("time").get("datetime"):
        published_at = soup.find("time").get("datetime")

    clean_text = soup.get_text(separator=" ")
    clean_text = " ".join(clean_text.split())
    clean_text = sanitize_for_embedding(clean_text)

    return title, clean_text, published_at


def extract_links(current_url: str, html: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for anchor in soup.find_all("a", href=True):
        joined = urljoin(current_url, anchor["href"])
        canonical = canonicalize_url(joined)
        if is_allowed_url(canonical):
            links.add(canonical)

    return links


def crawl_pages() -> list[dict]:
    run_time = utc_now_iso()
    queue = deque((canonicalize_url(seed), 0) for seed in cfg.SEED_URLS)
    queued = {canonicalize_url(seed) for seed in cfg.SEED_URLS}
    visited = set()

    crawled_pages: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        while queue and len(crawled_pages) < cfg.MAX_PAGES:
            url, depth = queue.popleft()
            if url in visited or depth > cfg.MAX_DEPTH or not is_allowed_url(url):
                continue

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=cfg.REQUEST_TIMEOUT_MS)
                page.wait_for_timeout(cfg.WAIT_AFTER_LOAD_MS)
                html = page.content()
            except Exception as exc:
                print(f"[WARN] Failed to load {url}: {exc}")
                visited.add(url)
                continue

            visited.add(url)

            title, clean_text, published_at = extract_page_fields(html)
            if clean_text:
                content_hash = sha256_text(clean_text)
                crawled_pages.append(
                    {
                        "url": url,
                        "depth": depth,
                        "title": title,
                        "published_at": published_at,
                        "clean_text": clean_text,
                        "content_hash": content_hash,
                        "page_type": classify_page_type(url),
                        "crawled_at": run_time,
                    }
                )

            if depth < cfg.MAX_DEPTH:
                for next_url in extract_links(url, html):
                    if next_url not in visited and next_url not in queued:
                        queue.append((next_url, depth + 1))
                        queued.add(next_url)

        context.close()
        browser.close()

    return crawled_pages


def chunk_page(page_record: dict) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )

    base_doc = Document(
        page_content=page_record["clean_text"],
        metadata={
            "source": page_record["url"],
            "url_canonical": page_record["url"],
            "title": page_record["title"] or "",
            "page_type": page_record["page_type"],
            "crawl_depth": page_record["depth"],
            "published_at": page_record["published_at"] or "",
            "content_hash": page_record["content_hash"],
            "crawled_at": page_record["crawled_at"],
        },
    )

    chunks = splitter.split_documents([base_doc])

    for chunk in chunks:
        chunk.page_content = sanitize_for_embedding(chunk.page_content)

    return [c for c in chunks if c.page_content.strip()]


def add_documents_with_isolation(
    vectorstore: Chroma,
    docs: list[Document],
    ids: list[str],
    failed_urls: set[str],
) -> tuple[int, int]:
    """
    Add documents with recursive fault isolation.
    Returns: (inserted_count, failed_count)
    """
    if not docs:
        return 0, 0

    try:
        vectorstore.add_documents(documents=docs, ids=ids)
        return len(docs), 0
    except Exception as exc:
        if len(docs) == 1:
            source = docs[0].metadata.get("source", "Unknown")
            failed_urls.add(source)
            print(f"[WARN] Skipping 1 chunk from {source} due to embedding error: {exc}")
            return 0, 1

        mid = len(docs) // 2
        left_ok, left_fail = add_documents_with_isolation(
            vectorstore,
            docs[:mid],
            ids[:mid],
            failed_urls,
        )
        right_ok, right_fail = add_documents_with_isolation(
            vectorstore,
            docs[mid:],
            ids[mid:],
            failed_urls,
        )
        return left_ok + right_ok, left_fail + right_fail


def mark_existing_as_not_current(collection, ids: list[str], metadatas: list[dict], last_seen_at: str) -> None:
    if not ids:
        return

    updated_metadatas = []
    for metadata in metadatas:
        updated = dict(metadata)
        updated["is_current"] = False
        updated["last_seen_at"] = last_seen_at
        updated_metadatas.append(updated)

    collection.update(ids=ids, metadatas=updated_metadatas)


def upsert_pages_to_vectorstore(pages: list[dict]) -> None:
    if not pages:
        print("No pages crawled. Nothing to ingest.")
        return

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name=cfg.COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=cfg.PERSIST_DIRECTORY,
    )

    collection = vectorstore._collection
    run_id = str(uuid.uuid4())
    now = utc_now_iso()

    docs_to_add: list[Document] = []
    ids_to_add: list[str] = []
    failed_urls: set[str] = set()

    skipped = 0
    inserted_pages = 0

    for page in pages:
        url = page["url"]
        content_hash = page["content_hash"]
        page_type = page["page_type"]

        existing = collection.get(where={"url_canonical": url}, include=["metadatas"])
        existing_ids = existing.get("ids", [])
        existing_metadatas = existing.get("metadatas", [])

        current_entries = [
            (doc_id, metadata)
            for doc_id, metadata in zip(existing_ids, existing_metadatas)
            if metadata.get("is_current", True)
        ]

        current_hash = current_entries[0][1].get("content_hash") if current_entries else None

        # Skip unchanged content to avoid duplicate embeddings.
        if current_hash == content_hash:
            skipped += 1
            continue

        if page_type == "mutable_live":
            # Mutable pages should keep only the latest snapshot active.
            if existing_ids:
                collection.delete(ids=existing_ids)
        else:
            # News pages retain history; old versions are marked non-current.
            if current_entries:
                mark_existing_as_not_current(
                    collection,
                    ids=[doc_id for doc_id, _ in current_entries],
                    metadatas=[metadata for _, metadata in current_entries],
                    last_seen_at=now,
                )

        page_version_id = sha256_text(f"{url}:{content_hash}")
        chunks = chunk_page(page)

        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "run_id": run_id,
                    "page_version_id": page_version_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "is_current": True,
                    "first_seen_at": now,
                    "last_seen_at": now,
                }
            )

            chunk_id = sha256_text(f"{page_version_id}:{idx}")
            docs_to_add.append(chunk)
            ids_to_add.append(chunk_id)

        inserted_pages += 1

    inserted_chunks = 0
    failed_chunks = 0
    if docs_to_add:
        inserted_chunks, failed_chunks = add_documents_with_isolation(
            vectorstore,
            docs_to_add,
            ids_to_add,
            failed_urls,
        )
git checkout -b pageindex-experiments

    if failed_urls:
        failure_payload = {
            "failed_url_count": len(failed_urls),
            "failed_urls": sorted(failed_urls),
        }
        with open("src/failed_ingest_urls.json", "w", encoding="utf-8") as fp:
            json.dump(failure_payload, fp, indent=2)
        print("[WARN] Some chunks failed embedding. Failed URLs written to src/failed_ingest_urls.json")
        for url in sorted(failed_urls):
            print(f"[WARN] Failed URL: {url}")

    print(
        f"Ingestion complete | crawled_pages={len(pages)} inserted_pages={inserted_pages} "
        f"skipped_unchanged={skipped} inserted_chunks={inserted_chunks} failed_chunks={failed_chunks}"
    )


if __name__ == "__main__":
    pages = crawl_pages()
    print(f"Crawled {len(pages)} pages inside IndiaToday T20 scope")
    upsert_pages_to_vectorstore(pages)