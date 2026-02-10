import os
import re

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import crawl_config as cfg



class RAGPipeline:
    def __init__(self):
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
            temperature=0.4,
            api_key=key,
        )

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
        docs = []

        if mode == "latest":
            docs = self.vectorstore.similarity_search(
                question,
                k=k,
                filter={"is_current": True},
            )
        elif mode == "historical":
            docs = self.vectorstore.similarity_search(question, k=k)
        else:
            # Balanced mode prefers current docs first, but falls back to full history.
            docs = self.vectorstore.similarity_search(
                question,
                k=k,
                filter={"is_current": True},
            )
            if not docs:
                docs = self.vectorstore.similarity_search(question, k=k)

        # Extra fallback for older collections that might not contain metadata filters.
        if not docs:
            docs = self.vectorstore.similarity_search(question, k=k)

        return docs

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
