import os
import time
from pageindex import PageIndexClient
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("PAGEINDEX_API_KEY")

if not API_KEY:
    raise ValueError("PAGEINDEX_API_KEY not set")

CHECKS = 3
INTERVAL = 5

client = PageIndexClient(api_key=API_KEY)


def snapshot(i):
    print(f"\n========== CHECK {i} ==========\n")

    try:
        result = client.list_documents()

        # SDK may return dict or list
        documents = result.get()

        print(f"Total docs stored: {len(documents)}")

        for d in documents[:5]:
            meta = d.get("metadata", {})
            print(
                f"- {meta.get('source_url')} | "
                f"hash={meta.get('content_hash')} | "
                f"time={meta.get('crawled_at')}"
            )

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    for i in range(1, CHECKS + 1):
        snapshot(i)
        if i < CHECKS:
            time.sleep(INTERVAL)
