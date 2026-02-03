# RAG with LangChain (Step-by-Step, From Scratch → Production)


* Document loading
* Chunking
* Embeddings
* Vector database
* Retrieval
* Prompt grounding
* LLM answer generation


---

# 🚀 Tech Stack

* **Python**
* **LangChain (core + community)**
* **Ollama (local LLM + embeddings)**
* **Chroma (vector database)**

Everything runs **fully local** (no API keys required).

---

# 🧠 RAG Architecture (Mental Model)

Manual RAG logic:

```
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector DB
   ↓
Top-k retrieval
   ↓
Prompt + LLM
   ↓
Grounded answer
```

LangChain mapping:

| Concept      | LangChain Component |
| ------------ | ------------------- |
| File loading | DocumentLoader      |
| Chunking     | TextSplitter        |
| Embeddings   | Embedding model     |
| Storage      | Chroma              |
| Retrieval    | Retriever           |
| Prompting    | PromptTemplate      |
| LLM          | ChatOllama          |

LangChain **does not change the algorithm**, it just standardizes the plumbing.

---

# 📦 Installation

## 1. Create virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Install Ollama

[https://ollama.com/download](https://ollama.com/download)

## 4. Pull models

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull phi3:mini         # small LLM
```

---

# 📁 Project Structure

```
Rag_LangChain/
│
├── data/
│   ├── langchain.txt
│   ├── rag.txt
│   └── llm.txt
│
├── src/
│   ├── step1_load_documents.py
│   ├── step2_chunk_documents.py
│   ├── step3_create_embeddings.py
│   ├── step4_vectorstore_chroma.py
│   └── step5_rag_pipeline.py
│
├── chroma_db/   (ignored)
├── requirements.txt
└── README.md
```

---

---

# ✅ Step 1 — Load Documents

## Goal

Convert raw files → LangChain `Document` objects.

## Why

Everything downstream operates on `Document`, not raw files.

## Code

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

print(documents[0].page_content)
print(documents[0].metadata)
```

## Run

```bash
python src/step1_load_documents.py
```

---

---

# ✅ Step 2 — Chunking

## Goal

Split long text into smaller semantic pieces.

## Why

* Better embeddings
* Better retrieval
* Avoid context limits

## Code

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
```

## Run

```bash
python src/step2_chunk_documents.py
```

---

---

# ✅ Step 3 — Embeddings (Ollama)

## Goal

Convert chunks → vectors.

## Why

Vector similarity powers retrieval.

## Code

```python
from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="nomic-embed-text")

vectors = emb.embed_documents([c.page_content for c in chunks])
print(len(vectors[0]))
```

## Run

```bash
python src/step3_create_embeddings.py
```

---

---

# ✅ Step 4 — Vector Database (Chroma)

## Goal

Store embeddings persistently and perform similarity search.

## Why

We need fast nearest-neighbor lookup.

## Code

```python
from langchain_chroma import Chroma

db = Chroma(
    collection_name="rag_demo",
    embedding_function=emb,
    persist_directory="./chroma_db"
)

db.add_documents(chunks)

results = db.similarity_search("What is RAG?", k=2)
```

## Run

```bash
python src/step4_vectorstore_chroma.py
```

---

---

# ✅ Step 5 — Full RAG Pipeline

## Goal

Retrieve context → prompt LLM → grounded answer.

## Code

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

retriever = db.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="phi3:mini", temperature=0)

docs = retriever.invoke("What is RAG?")
context = "\n\n".join(d.page_content for d in docs)

prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
What is RAG?
"""

print(llm.invoke(prompt))
```

## Run

```bash
python src/step5_rag_pipeline.py
```

---

---

# 🧹 Important Notes

## Ignore generated data

Add to `.gitignore`:

```
chroma_db/
.venv/
```

## Stop running models

```
ollama ps
ollama stop all
```

## Lightweight models

```
phi3:mini (~2GB)
gemma:2b (~1.7GB)
qwen2:3b (~3GB)
```

---

---

# 🎯 What You Learn From This Repo

By completing all steps you understand:

✅ Document ingestion
✅ Chunking strategies
✅ Embeddings
✅ Vector DB internals
✅ Retrieval
✅ Prompt grounding
✅ Local LLM inference
✅ Full production-style RAG

Not just “copy-paste chains”.

---

# 🚀 Next Improvements

Possible upgrades:

* PDF ingestion
* metadata filtering
* source citations
* re-ranking
* streaming responses
* evaluation metrics
* deployment API

---

# 📌 Philosophy

> Learn the plumbing first.
> Then use abstractions.

This repo builds RAG **bottom-up**, the same way real systems are engineered.

---

---

# 🚀 Backend Service + Streamlit Frontend

After completing the step-by-step RAG pipeline, the project was refactored into a cleaner **backend + frontend architecture** so the system behaves like a real application instead of standalone scripts.

Instead of running separate scripts for each step, we now expose the RAG logic through a reusable backend service and interact with it through a Streamlit UI.

## 🧠 Architecture

```
Streamlit UI (app.py)
        ↓
RAG Backend Service (pipeline.py)
        ↓
Chroma Vector DB + Ollama (Embeddings + LLM)
```

The **frontend handles only UI**, while all **RAG logic lives in the backend**.
This separation mirrors how production AI systems are built.

---

## 📦 pipeline.py — Backend (RAG Service)

This file contains the **core RAG engine** and is responsible for:

* Initializing embeddings (`OllamaEmbeddings`)
* Loading persistent Chroma DB
* Creating retriever (`top-k similarity search`)
* Initializing LLM (`ChatOllama`)
* Defining a reusable `ask(question)` method

All heavy components (models + DB) are loaded **once during initialization**, which makes queries fast and avoids reloading models on every request.

### Usage

```python
from src.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.ask("What is RAG?")
print(result["answer"])
print(result["sources"])
```

### Why this design?

* avoids re-embedding data repeatedly
* avoids reloading LLM per query
* faster responses
* reusable from any UI/API
* production-style backend service pattern

---

## 🎨 app.py — Frontend (Streamlit UI)

This file provides a simple **web interface** to interact with the RAG backend.

Features:

* Chat-style interface
* User question input
* Displays answers
* Shows document sources
* Uses `@st.cache_resource` to load backend only once
* Runs fully local (no API keys required)

### Run the app

```bash
streamlit run app.py
```

Then open the browser and start chatting with your documents.

---

## ✅ Result

With these two files, the project now supports:

* Modular backend service
* Interactive frontend
* Persistent vector database
* Local embeddings + LLM
* Real application structure (not just scripts)

## This completes the transition from **learning each RAG component → building a deployable RAG application**.

