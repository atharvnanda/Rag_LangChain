from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from bs4 import BeautifulSoup

# "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
# "https://docs.langchain.com/",
# "https://docs.python.org/3/"
# "example.org"

#"https://www.indiatoday.in/business/market/story/anthropic-new-ai-tools-panic-on-wall-street-it-stocks-hit-explained-2862772-2026-02-04"
URLS = [
    "https://www.indiatoday.in/"
]

def load_urls(URLS):
    loader = PlaywrightURLLoader(
        urls=URLS,
        headless=True,
        remove_selectors=["script", "style", "nav", "footer"]
    )
    return loader.load()

def clean_documents(docs):
    cleaned_docs = []

    for doc in docs:
        soup = BeautifulSoup(doc.page_content, "html.parser")

        # remove useless tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # extract only meaningful text blocks
        

        clean_text = soup.get_text()
        clean_text = ' '.join(clean_text.strip().split())

        if clean_text:
            doc.page_content = clean_text
            cleaned_docs.append(doc)
    return cleaned_docs


def chunking(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )

    cks = text_splitter.split_documents(docs)

    clean_chunks = [ 
        c for c in cks 
        if c.page_content.strip() 
    ]

    return clean_chunks

def embedStore(data):
    embedding_model = OllamaEmbeddings(
        model = "nomic-embed-text"
    )

    vectorstore = Chroma(
        collection_name = "rag_it",
        embedding_function = embedding_model,
        persist_directory = "./chroma_db"
    )

    vectorstore.add_documents(data)

if __name__ == "__main__":
    #loading data
    url_docs = load_urls(URLS)
    url_docs = clean_documents(url_docs)
    print(f"Loaded {len(url_docs)} Documents")

    #chunking
    chunks = chunking(url_docs)
    print(f"Loaded {len(chunks)} Chunks")

    #adding documents to vectorstore (embedding + storing)
    embedStore(chunks)
    print("Documents stored in Chroma")