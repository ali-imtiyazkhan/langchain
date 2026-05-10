from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load the existing vector store
print("Loading vector store (FAISS)...")
vector_store = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Use the vector store as an MMR retriever
# MMR (Maximum Marginal Relevance) tries to balance relevance and diversity.
# fetch_k: Number of documents to fetch to pass to MMR algorithm (default 20)
# k: Number of documents to return after MMR (default 4)
# lambda_mult: Number between 0 and 1 that determines the degree of diversity among the results.
# 0 corresponds to maximum diversity and 1 to minimum diversity (default 0.5)
print("Initializing MMR Retriever...")
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Perform retrieval
query = "What is LangChain?"
print(f"Retrieving diverse relevant chunks for: '{query}'...")

try:
    docs = retriever.invoke(query)

    print(f"\n--- Retrieved {len(docs)} documents using MMR ---")

    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

except Exception as e:
    print(f"Error: {e}")
