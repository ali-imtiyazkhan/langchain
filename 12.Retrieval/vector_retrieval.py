import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Load the existing vector store
print("Loading vector store (FAISS)...")
vector_store = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. Use the vector store as a retriever
# search_type can be "similarity", "mmr" (Maximum Marginal Relevance), etc.
# search_kwargs allows specifying 'k' (number of results) and other parameters
print("Initializing Vector Store Retriever...")
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# 4. Perform retrieval
query = "What is LangChain?"
print(f"Retrieving relevant chunks for: '{query}'...")

try:
    docs = retriever.invoke(query)

    print(f"\n--- Retrieved {len(docs)} documents ---")

    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

except Exception as e:
    print(f"Error: {e}")
