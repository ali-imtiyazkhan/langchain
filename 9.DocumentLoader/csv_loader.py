# pyrefly: ignore [missing-import]
from langchain_community.document_loaders.csv_loader import CSVLoader

# You can also specify the metadata_columns if you want specific columns in metadata
loader = CSVLoader(file_path="example.csv")

documents = loader.load()

print(f"--- Loaded {len(documents)} documents from CSV ---")
for doc in documents:
    print("-" * 20)
    print(f"Metadata: {doc.metadata}")
    print(f"Content:\n{doc.page_content}")
