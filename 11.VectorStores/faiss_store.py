import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

# 1. Load and Chunk the data
loader = TextLoader("example.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 2. Initialize the Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. Create the Vector Store and Store Chunks
print("Creating vector store...")
vector_store = FAISS.from_documents(docs, embeddings)

# 4. Save the vector store locally
vector_store.save_local("faiss_index")
print("Vector store saved to 'faiss_index' folder.")

# 5. Search Example
query = "What is LangChain?"
results = vector_store.similarity_search(query, k=2)

print(f"\n--- Search results for: '{query}' ---")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
