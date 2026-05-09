# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader('../', glob='*.pdf',loader_cls=PyPDFLoader, recursive=True, use_multithreading=True)
documents = loader.load()
print(documents)