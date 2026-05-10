import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

# 1. Load and Chunk the data
# Ensure you have an 'example.txt' file in the root directory
file_path = os.path.join(os.path.dirname(__file__), "..", "example.txt")
loader = TextLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 2. Initialize the Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. Create the Vector Store and Store Chunks
print("Creating Chroma vector store...")
persist_directory = "./chroma_db"
vector_store = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"Vector store saved to '{persist_directory}' folder.")

# 4. Search Example
query = "What is LangChain?"
results = vector_store.similarity_search(query, k=2)

print(f"\n--- Search results for: '{query}' ---")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
