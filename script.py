from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

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

embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

print(f'Generated {len(embeddings)} embeddings.')
print(embeddings[0][:5]) 