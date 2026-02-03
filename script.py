from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

#loading data
loader = DirectoryLoader('./data', glob='**/*.txt', loader_cls=TextLoader)

documents = loader.load()

print(f'Loaded {len(documents)} documents.')

for i, doc in enumerate(documents):
    print(f'Document {i+1} content:\n{doc.page_content}\n metadata: {doc.metadata}\n')

#chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50
)

chunks = text_splitter.split_documents(documents)

for i,doc in enumerate(chunks[:5]):
    print(f'Chunk {i+1} content:\n{doc.page_content}\n metadata: {doc.metadata}\n')

#embedding

embedding_model = OllamaEmbeddings(
    model = "nomic-embed-text"
)

#creating a vectorstore

vectorstore = Chroma(
    collection_name = "rag_demo",
    embedding_function = embedding_model,
    persist_directory = './chroma_db'
)

#adding documents to vectorstore (embedding + storing)
if vectorstore._collection.count() == 0:
    vectorstore.add_documents(chunks)

print("Documents stored in Chroma")

#retriever function

retriever = vectorstore.as_retriever(
    search_kwargs = { #
        "k": 2
    }
)

#LLM

llm = ChatOllama(
    model = "mistral",
    temparature = 0 #
)

#prompt template

prompt_template = PromptTemplate(
    input_variables = ["context", "question"],
    template = """You are a helpful assistant. Use the following context to answer the question. If you don't know the answer, say you don't know.
    Context:
{context}

Question:
{question}

Answer:
"""
)

#RAG function

query = "What is RAG?"

retrieved_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in retrieved_docs])

prompt = prompt_template.format(
    context=context, 
    question=query
)

response = llm.invoke(prompt)

print(f"Question: {query}\n")
print(f"Answer: {response.content}\n")
