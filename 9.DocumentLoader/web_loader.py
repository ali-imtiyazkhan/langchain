# WebBaseLoader requires beautifulsoup4 to be installed: pip install beautifulsoup4
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import WebBaseLoader

# 1. Initialize the loader with a URL
url = "https://python.langchain.com/v0.2/docs/introduction/"
loader = WebBaseLoader(url)

# 2. Load the content
try:
    documents = loader.load()

    # 3. Print the content
    print(f"--- Content from {url} ---")
    print(f"Title: {documents[0].metadata.get('title', 'No Title')}")
    print(f"Content Preview: {documents[0].page_content[:500]}...")
    
except Exception as e:
    print(f"Error: {e}")
    print("Tip: Ensure 'beautifulsoup4' is installed (pip install beautifulsoup4)")
