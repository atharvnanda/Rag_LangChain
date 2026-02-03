from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader('./data', glob='**/*.txt', loader_cls=TextLoader)

documents = loader.load()

print(f'Loaded {len(documents)} documents.')

for i, doc in enumerate(documents):
    print(f'Document {i+1} content:\n{doc.page_content}\n metadata: {doc.metadata}\n')