import os
import re

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import crawl_config as cfg


class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN"],
        )
        self.model = model_name

    def embed_documents(self, texts):
        batch_size = 32
        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(
                self.client.feature_extraction(texts[i:i+batch_size], model=self.model)
            )
        return results

    def embed_query(self, text):
        return self.client.feature_extraction(text, model=self.model)

class RAGPipeline:
    def __init__(self):
        load_dotenv()
        # embeddings
        self.embedding_model = HFInferenceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )

        # vector db
        self.vectorstore = Chroma(
            collection_name=cfg.COLLECTION_NAME,  # prev vector db : rag_demo , rag_web, rag_it, rag_t20
            embedding_function=self.embedding_model,
            persist_directory=cfg.PERSIST_DIRECTORY,
        )

        # llm
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


    def _retrieve_documents(self, question: str, k: int = 5):
        return self.vectorstore.similarity_search(question, k=k)

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
