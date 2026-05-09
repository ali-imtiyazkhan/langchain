# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import TextLoader

# Path to the test file
file_path = "example.txt"

# Initialize the loader
loader = TextLoader(file_path)

print("--- Using load() ---")
# load() returns a list of all documents
documents = loader.load()
print(f"Type of result: {type(documents)}")
for doc in documents:
    print(f"Content: {doc.page_content[:50]}...")

print("\n--- Using lazy_load() ---")
# lazy_load() returns a generator (iterator)
lazy_documents = loader.lazy_load()
print(f"Type of result: {type(lazy_documents)}")

for doc in lazy_documents:
    print(f"Yielded Content: {doc.page_content[:50]}...")

# Another way to consume lazy_load
# next_doc = next(loader.lazy_load())
# print(f"First doc from generator: {next_doc.page_content[:50]}...")
