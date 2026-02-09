import asyncio
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler

OUTPUT_DIR = Path("data/raw")

URLS_TO_CRAWL = [
]

async def crawl_site(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)

    markdown = result.markdown
    html = result.html
    links = result.links

    return {
        "url": url,
        "markdown": markdown,
        "html": html,
        "links": links,
    }


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [crawl_site(url) for url in URLS_TO_CRAWL]
    pages = await asyncio.gather(*tasks)

    for page in pages:
        slug = page["url"].replace("://", "_").replace("/", "_")
        out_file = OUTPUT_DIR / f"{slug}.json"
        with open(out_file, "w") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)

    print(f"Crawled {len(pages)} pages.")

if __name__ == "__main__":
    asyncio.run(main())
