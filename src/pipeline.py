import os
import re
from hashlib import sha256

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import crawl_config as cfg



class RAGPipeline:
    def __init__(self):
        self.debug = os.getenv("RAG_DEBUG", "0") == "1"

        # embeddings
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # vector db
        self.vectorstore = Chroma(
            collection_name=cfg.COLLECTION_NAME,  # prev vector db : rag_demo , rag_web, rag_it
            embedding_function=self.embedding_model,
            persist_directory=cfg.PERSIST_DIRECTORY,
        )

        # llm
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=key,
        )

        # lexical retrievers (BM25)
        self.bm25_all, self.bm25_current = self._build_bm25_retrievers()

        # prompt
        self.prompt = PromptTemplate(
            input_variables=["history", "context", "question", "mode"],
            template="""
You are an intelligent, conversational AI assistant focused ONLY on ICC T20 World Cup information sourced from IndiaToday pages provided by retrieval.
Use ONLY the context. Do not use outside knowledge.
If answer is not present in context, strictly say: "I apologise! I don't know".
Use headings and bullets when helpful.

Retrieval mode:
{mode}

History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
        )

    @staticmethod
    def _normalize_query(question: str) -> str:
        q = question.lower().strip()
        q = re.sub(r"[^a-z0-9\s]", " ", q)
        q = re.sub(r"\s+", " ", q)
        return q

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _doc_key(doc: Document) -> str:
        source = doc.metadata.get("source", "")
        page_version = doc.metadata.get("page_version_id", "")
        chunk_index = str(doc.metadata.get("chunk_index", ""))
        raw = f"{source}|{page_version}|{chunk_index}|{doc.page_content[:120]}"
        return sha256(raw.encode("utf-8")).hexdigest()

    def _build_bm25_retrievers(self):
        payload = self.vectorstore.get(include=["documents", "metadatas"])
        documents = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []

        docs_all: list[Document] = []
        docs_current: list[Document] = []

        for text, meta in zip(documents, metadatas):
            metadata = meta or {}
            doc = Document(page_content=text, metadata=metadata)
            docs_all.append(doc)
            if metadata.get("is_current", True):
                docs_current.append(doc)

        bm25_all = BM25Retriever.from_documents(docs_all) if docs_all else None
        bm25_current = BM25Retriever.from_documents(docs_current) if docs_current else None

        if bm25_all:
            bm25_all.k = 10
        if bm25_current:
            bm25_current.k = 10

        return bm25_all, bm25_current

    def _bm25_search(self, question: str, mode: str, k: int = 8) -> list[Document]:
        retriever = self.bm25_current if mode in {"latest", "balanced"} else self.bm25_all
        if retriever is None:
            return []
        retriever.k = k
        return retriever.invoke(question)

    def _debug_hits(self, label: str, docs: list[Document]) -> None:
        if not self.debug:
            return
        print(f"\n[DEBUG] {label} ({len(docs)} docs)")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            snippet = " ".join(doc.page_content.split())[:180]
            print(f"  {i}. {source} | {snippet}")

    def _merge_and_rerank(
        self,
        question: str,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        final_k: int = 5,
    ) -> list[Document]:
        norm_q = self._normalize_query(question)
        q_tokens = self._token_set(norm_q)

        score_map = {}

        for rank, doc in enumerate(vector_docs, start=1):
            key = self._doc_key(doc)
            score_map.setdefault(key, {"doc": doc, "vec": 0.0, "bm25": 0.0, "lex": 0.0})
            score_map[key]["vec"] = max(score_map[key]["vec"], 1.0 / rank)

        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._doc_key(doc)
            score_map.setdefault(key, {"doc": doc, "vec": 0.0, "bm25": 0.0, "lex": 0.0})
            score_map[key]["bm25"] = max(score_map[key]["bm25"], 1.0 / rank)

        for key, row in score_map.items():
            doc_tokens = self._token_set(row["doc"].page_content[:1200])
            if q_tokens:
                row["lex"] = len(q_tokens.intersection(doc_tokens)) / max(len(q_tokens), 1)
            else:
                row["lex"] = 0.0

        ranked = sorted(
            score_map.values(),
            key=lambda x: 0.45 * x["vec"] + 0.40 * x["bm25"] + 0.15 * x["lex"],
            reverse=True,
        )

        docs = [row["doc"] for row in ranked[:final_k]]
        self._debug_hits("Final reranked", docs)
        return docs

    @staticmethod
    def _detect_query_mode(question: str) -> str:
        q = question.lower()

        latest_patterns = [
            r"\blatest\b",
            r"\bcurrent\b",
            r"\btoday\b",
            r"\bnow\b",
            r"\brecent\b",
            r"\bpoints table\b",
            r"\bschedule\b",
            r"\bstandings\b",
            r"\bscoreboard\b",
        ]
        historical_patterns = [
            r"\bold\b",
            r"\bhistorical\b",
            r"\bprevious\b",
            r"\bearlier\b",
            r"\bpast\b",
            r"\barchive\b",
            r"\blast year\b",
        ]

        if any(re.search(pattern, q) for pattern in latest_patterns):
            return "latest"
        if any(re.search(pattern, q) for pattern in historical_patterns):
            return "historical"
        return "balanced"

    def _retrieve_documents(self, question: str, mode: str, k: int = 5):
        norm_question = self._normalize_query(question)
        vector_docs = []

        if mode == "latest":
            vector_docs = self.vectorstore.similarity_search(
                norm_question,
                k=k,
                filter={"is_current": True},
            )
        elif mode == "historical":
            vector_docs = self.vectorstore.similarity_search(norm_question, k=k)
        else:
            # Balanced mode prefers current docs first, but falls back to full history.
            vector_docs = self.vectorstore.similarity_search(
                norm_question,
                k=k,
                filter={"is_current": True},
            )
            if not vector_docs:
                vector_docs = self.vectorstore.similarity_search(norm_question, k=k)

        # Extra fallback for older collections that might not contain metadata filters.
        if not vector_docs:
            vector_docs = self.vectorstore.similarity_search(norm_question, k=k)

        bm25_docs = self._bm25_search(norm_question, mode, k=max(8, k))

        self._debug_hits("Vector hits", vector_docs)
        self._debug_hits("BM25 hits", bm25_docs)

        return self._merge_and_rerank(norm_question, vector_docs, bm25_docs, final_k=k)

    def ask(self, question: str, chat_history: list):
        mode = self._detect_query_mode(question)
        docs = self._retrieve_documents(question, mode)

        context = "\n\n".join(d.page_content for d in docs)

        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in chat_history[-6:]
        )

        prompt = self.prompt.format(
            history=history_text,
            context=context,
            question=question,
            mode=mode,
        )

        response = self.llm.invoke(prompt)

        sources = list({
            d.metadata.get("source", "Unknown")
            for d in docs
        })

        return response.content, sources, mode
