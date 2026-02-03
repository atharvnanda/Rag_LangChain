from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


class RAGPipeline:
    def __init__(self):
        # embeddings
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # vector db
        self.vectorstore = Chroma(
            collection_name="rag_demo",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"
        )

        # retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        # llm
        self.llm = ChatOllama(model="llama3.2:latest", temperature=1)

        # prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Answer ONLY from the provided context.
If not found, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def ask(self, question: str):
        docs = self.retriever.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = self.prompt.format(context=context, question=question)

        response = self.llm.invoke(prompt)

        return response.content, docs
