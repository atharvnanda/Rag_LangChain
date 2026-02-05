from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os



class RAGPipeline:
    def __init__(self):
        # embeddings
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # vector db
        self.vectorstore = Chroma(
            collection_name="rag_it", #prev vector db : rag_demo , rag_web
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"
        )

        # retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # llm
        # self.llm = ChatOllama(model="llama3.2:latest", temperature=1)
        
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.5,
            api_key=key
        )

        # prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""

Answer from the BOTH provided context AND chat history (to retain chat memory) to give news and provide facts . Can be slightly conversational with user.
USE HEADINGS, BULLETS, EMPHASIS to make the answer more readable and structured.
If the question cannot be answered from the context, just say "I don't know".

History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def ask(self, question: str, chat_history: list):
        docs = self.retriever.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)

        history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in chat_history[-6:]
        )

        prompt = self.prompt.format(history=history_text,context=context, question=question)

        response = self.llm.invoke(prompt)

        return response.content, docs
