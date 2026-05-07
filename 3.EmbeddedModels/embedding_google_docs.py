from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the embedding model
# Note: Ensure GOOGLE_API_KEY is set in your .env file
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# List of documents to embed
documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of India is New Delhi.",
    "The capital of Japan is Tokyo."
]

# Embed the documents
print("Embedding documents...")
doc_embeddings = embeddings.embed_documents(documents)
print(f"Successfully embedded {len(doc_embeddings)} documents.")
print(f"Dimension of each embedding: {len(doc_embeddings[0])}")

# Embed a query
query = "What is the capital of India?"
print(f"\nEmbedding query: '{query}'")
query_embedding = embeddings.embed_query(query)
print(f"Successfully embedded the query.")
print(f"Dimension of query embedding: {len(query_embedding)}")

# Optional: You could now use these embeddings for semantic search or similarity comparison
