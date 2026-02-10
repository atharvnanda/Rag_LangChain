import os
import time
import hashlib
from pathlib import Path
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from pageindex import PageIndexClient
import crawl_config as cfg


# CONFIG

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MARKDOWN_FILE = DATA_DIR / "site_dump.md"
DOC_ID_FILE = DATA_DIR / "doc_id.txt"


# Helpers

def sha256(text: str):
    return hashlib.sha256(text.encode()).hexdigest()


def canonicalize(url):
    p = urlparse(url)
    return urlunparse(("https", p.netloc.lower(), p.path.rstrip("/") or "/", "", "", ""))


def allowed(url):
    if any(d in url for d in cfg.DENY_URL_SUBSTRINGS):
        return False

    parsed = urlparse(url)

    return (
        parsed.netloc in cfg.ALLOWED_DOMAINS
        and any(prefix in parsed.path for prefix in cfg.ALLOWED_PATH_PREFIXES)
    )


# Crawl site

def crawl_pages():
    queue = deque((canonicalize(seed), 0) for seed in cfg.SEED_URLS)
    visited = set()
    pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        while queue and len(visited) < cfg.MAX_PAGES:
            url, depth = queue.popleft()

            if url in visited or depth > cfg.MAX_DEPTH or not allowed(url):
                continue

            visited.add(url)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=cfg.REQUEST_TIMEOUT_MS)
                page.wait_for_timeout(cfg.WAIT_AFTER_LOAD_MS)
                html = page.content()
            except Exception:
                continue

            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = " ".join(soup.get_text().split())

            if text:
                pages.append((url, text))

            if depth < cfg.MAX_DEPTH:
                for a in soup.find_all("a", href=True):
                    nxt = canonicalize(urljoin(url, a["href"]))
                    if allowed(nxt):
                        queue.append((nxt, depth + 1))

        browser.close()

    return pages


# Markdown handling

def load_existing_hashes():
    if not MARKDOWN_FILE.exists():
        return set()

    hashes = set()

    with open(MARKDOWN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("<!--hash:"):
                hashes.add(line.strip()[9:-4])

    return hashes


def append_new_pages(pages):
    existing_hashes = load_existing_hashes()

    new_sections = []

    for url, text in pages:
        h = sha256(text)

        if h in existing_hashes:
            continue

        section = f"\n\n<!--hash:{h}-->\n# {url}\n\n{text}\n"
        new_sections.append(section)

    if new_sections:
        with open(MARKDOWN_FILE, "a", encoding="utf-8") as f:
            f.writelines(new_sections)

    print(f"Appended {len(new_sections)} new/changed pages")


# PageIndex upload

def upload_tree():
    client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))

    print("Submitting markdown to PageIndex...")

    res = client.submit_document(str(MARKDOWN_FILE))
    new_doc_id = res["doc_id"]

    print("New doc_id:", new_doc_id)

    # wait for processing
    while True:
        status = client.get_document(new_doc_id)["status"]
        print("status:", status)

        if status == "completed":
            break

        time.sleep(3)

    # delete old tree
    if DOC_ID_FILE.exists():
        old_id = DOC_ID_FILE.read_text().strip()
        if old_id:
            print("Deleting old tree:", old_id)
            client.delete_document(old_id)

    DOC_ID_FILE.write_text(new_doc_id)


# Main

if __name__ == "__main__":
    load_dotenv()
    pages = crawl_pages()
    print(f"Crawled {len(pages)} pages")

    append_new_pages(pages)
    upload_tree()
