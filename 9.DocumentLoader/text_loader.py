# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import TextLoader

loader = TextLoader("example.txt")
documents = loader.load()
print(documents[0].page_content)