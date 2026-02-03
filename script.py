from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

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

vectorstore.add_documents(chunks)

print("Documents stored in Chroma")

#similarity search

query = "What is RAG?"
results = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f'Result {i+1} content:\n{doc.page_content}\n metadata: {doc.metadata}\n')