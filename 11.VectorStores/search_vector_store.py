# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the same embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Load the existing vector store from the local folder
print("Loading vector store...")
vector_store = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. Perform a similarity search
# Try changing the query!
query = input("\nEnter your search query: ")
results = vector_store.similarity_search(query, k=2)

print(f"\n--- Search results for: '{query}' ---")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
    print(f"Source: {doc.metadata.get('source')}")
