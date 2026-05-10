import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()

# 1. Initialize the Wikipedia Retriever
# top_k_results: Number of documents to retrieve
print("Initializing Wikipedia Retriever...")
retriever = WikipediaRetriever(top_k_results=3, lang="en")

# 2. Perform a retrieval
query = "What is LangChain?"
print(f"Searching Wikipedia for: '{query}'...")

try:
    # get_relevant_documents is the standard method for retrievers
    docs = retriever.invoke(query)

    print(f"\n--- Retrieved {len(docs)} documents from Wikipedia ---")

    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Summary Preview: {doc.page_content[:300]}...")

except Exception as e:
    print(f"Error: {e}")
    print("Tip: Ensure 'wikipedia' is installed (pip install wikipedia)")
