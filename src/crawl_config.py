"""Configuration for IndiaToday T20 World Cup crawling + ingestion."""

SEED_URLS = [
    "https://www.indiatoday.in/sports/cricket/t20-world-cup",
    "https://www.indiatoday.in/sports/cricket/t20-world-cup/schedule",
    "https://www.indiatoday.in/sports/cricket/t20-world-cup/points-table",
    "https://www.indiatoday.in/sports/cricket/videos"
]

# Strict domain/path controls so crawler never drifts to unrelated websites/topics.
ALLOWED_DOMAINS = {"www.indiatoday.in"}
ALLOWED_PATH_PREFIXES = ["/sports/cricket/"]

# Common URL fragments to ignore.
DENY_URL_SUBSTRINGS = [
    "#",
    "javascript:",
    "mailto:",
    "tel:",
    "/video",
    "/photos",
    "/live-tv",
    "/podcast",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
]

MUTABLE_PAGE_HINTS = [
    "schedule",
    "points-table",
    "scoreboard",
    "score-card",
    "live-score",
]

# Crawl controls
MAX_DEPTH = 2
MAX_PAGES = 60
REQUEST_TIMEOUT_MS = 15000
WAIT_AFTER_LOAD_MS = 1000

# Chunking + storage
COLLECTION_NAME = "rag_t20_new"
PERSIST_DIRECTORY = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
