# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('lec1.pdf')
documents = loader.load()
print(documents)