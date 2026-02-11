from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import unicodedata
import uuid
from urllib.parse import urljoin, urlparse, urlunparse
import os
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import crawl_config as cfg
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

class LocalQwenEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                         device="cuda")

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


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
    '''Basic URL canonicalization for deduplication and stability.'''
    parsed = urlparse(url)
    scheme = "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", "", ""))


def is_allowed_url(url: str) -> bool:
    '''Checks if a URL is within the allowed scope based on domain, path, and deny substrings.'''
    lowered = url.lower()
    if any(deny in lowered for deny in cfg.DENY_URL_SUBSTRINGS):
        return False

    parsed = urlparse(url)
    if parsed.netloc.lower() not in cfg.ALLOWED_DOMAINS:
        return False

    return any(parsed.path.startswith(prefix) for prefix in cfg.ALLOWED_PATH_PREFIXES)

def extract_page_fields(html: str) -> tuple[str, str, str | None]:
    '''Extracts title, clean text, and published date from HTML content.'''
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
    '''Extracts and canonicalizes all links from the given HTML content.'''
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for anchor in soup.find_all("a", href=True):
        joined = urljoin(current_url, anchor["href"])
        canonical = canonicalize_url(joined)
        if is_allowed_url(canonical):
            links.add(canonical)

    return links


def crawl_pages() -> list[dict]:
    '''Crawls pages starting from seed URLs, respecting depth and page limits, and extracts relevant fields.'''
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
                        "html": html,
                        "content_hash": content_hash,
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
    '''Converts a crawled page record into a list of Document chunks suitable for embedding and storage.'''
    soup = BeautifulSoup(page_record["html"], "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg"]):
        tag.decompose()


    blocks = []
    current_block = []

    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        if tag.name in ["h1", "h2", "h3"]:
            if current_block:
                blocks.append(" ".join(current_block))
                current_block = []
            current_block.append(text)
        else:
            current_block.append(text)

    if current_block:
        blocks.append(" ".join(current_block))

    chunks = []
    for block in blocks:
        block = sanitize_for_embedding(block)

        if len(block) < 200:
            if chunks:
                chunks[-1] += " " + block
            else:
                chunks.append(block)
        elif len(block) > 1200:
            sentences = sent_tokenize(block)
            temp = ""
            for s in sentences:
                if len(temp) + len(s) > 800:
                    chunks.append(temp.strip())
                    temp = s
                else:
                    temp += " " + s
            if temp:
                chunks.append(temp.strip())
        else:
            chunks.append(block)

    documents = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": page_record["url"],
                    "url_canonical": page_record["url"],
                    "title": page_record["title"] or "",
                    "crawl_depth": page_record["depth"],
                    "published_at": page_record["published_at"] or "",
                    "content_hash": page_record["content_hash"],
                    "crawled_at": page_record["crawled_at"],
                }
            )
        )

    return documents


def add_documents_with_isolation(
    vectorstore: Chroma,
    docs: list[Document],
    ids: list[str],
    failed_urls: set[str],
) -> tuple[int, int]:
    '''Adds documents to the vectorstore with error handling and isolation.'''

    if not docs:
        return 0, 0

    try:
        vectorstore.add_documents(documents=docs, ids=ids)
        print(f"Added {len(docs)} chunks for URL: {docs[0].metadata.get('source', 'Unknown')}")
        return len(docs), 0
    except Exception as exc:
        if len(docs) == 1:
            source = docs[0].metadata.get("source", "Unknown")
            failed_urls.add(source)
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

def upsert_pages_to_vectorstore(pages: list[dict]) -> None:
    '''Ingests crawled pages into the vectorstore.'''
    if not pages:
        print("No pages crawled. Nothing to ingest.")
        return
    
    

    embedding_model = LocalQwenEmbeddings()

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

        existing = collection.get(where={"url_canonical": url}, include=["metadatas"])
        existing_metadatas = existing.get("metadatas", [])

        # ---- SIMPLE DEDUPE ----
        if existing_metadatas:
            existing_hash = existing_metadatas[0].get("content_hash")
            if existing_hash == content_hash:
                skipped += 1
                continue

        page_version_id = sha256_text(f"{url}:{content_hash}")
        chunks = chunk_page(page)
        print(f"{page['url']} → {len(chunks)} chunks")


        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "run_id": run_id,
                    "page_version_id": page_version_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
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

    # load_dotenv()
    # client = InferenceClient(
    #     provider="hf-inference",
    #     api_key=os.environ["HF_TOKEN"],
    # )

    pages = crawl_pages()
    print(f"Crawled {len(pages)} pages inside IndiaToday T20 scope")
    upsert_pages_to_vectorstore(pages)