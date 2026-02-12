import os
import re

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

import crawl_config as cfg

class LocalQwenEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                         device="cuda",
                                         trust_remote_code=True)

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

class RAGPipeline:
    def __init__(self):
        load_dotenv()
        # embeddings
        self.embedding_model = LocalQwenEmbeddings()

        # vector db
        self.vectorstore = Chroma(
            collection_name=cfg.COLLECTION_NAME,  # prev vector db : rag_demo , rag_web, rag_it, rag_t20
            embedding_function=self.embedding_model,
            persist_directory=cfg.PERSIST_DIRECTORY,
        )

        # dense retriever
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        # load all docs once for BM25
        all_data = self.vectorstore.get()
        all_docs = all_data["documents"]
        metas = all_data["metadatas"]

        from langchain_core.documents import Document
        docs_for_bm25 = [
            Document(page_content=t, metadata=m)
            for t, m in zip(all_docs, metas)
        ]

        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = 10

        # hybrid retriever (RRF fusion)
        self.retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )

        self.reranker = CrossEncoder(
            "Qwen/Qwen3-Reranker-0.6B",
            device="cuda",
            trust_remote_code=True
        )

        if self.reranker.tokenizer.pad_token is None:
            self.reranker.tokenizer.pad_token = self.reranker.tokenizer.eos_token

        # llm
        key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            api_key=key,
        )

        # prompt
        self.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template="""
You are an intelligent, conversational AI assistant focused ONLY on ICC T20 World Cup information sourced from IndiaToday pages provided by retrieval.
Use ONLY the context. Do not use outside knowledge.
If answer is not present in context, strictly say: "I apologise! I don't know".
Use headings and bullets when helpful.

History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
        )


    def _retrieve_documents(self, question: str, k: int = 5):
        # fetch larger candidate pool
        docs = self.retriever.invoke(question)

        # prepare pairs for cross encoder
        pairs = [(question, d.page_content) for d in docs]

        scores = self.reranker.predict(pairs, batch_size=1)

        # sort by score descending
        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        # return top-k
        return [doc for _, doc in ranked[:k]]


    def ask(self, question: str, chat_history: list):
        docs = self._retrieve_documents(question)

        context = "\n\n".join(d.page_content for d in docs)

        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in chat_history[-6:]
        )

        prompt = self.prompt.format(
            history=history_text,
            context=context,
            question=question,
        )

        response = self.llm.invoke(prompt)

        sources = list({
            d.metadata.get("source", "Unknown")
            for d in docs
        })

        return response.content, sources
