import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import GithubFileLoader

load_dotenv()

# To use GitHub Loader, you typically need a Personal Access Token (PAT)
# Set GITHUB_PERSONAL_ACCESS_TOKEN in your .env file
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

if not access_token:
    print("Warning: GITHUB_PERSONAL_ACCESS_TOKEN not found in environment.")
    print("For public repos, it might still work, but you'll hit rate limits quickly.")

# Example: Loading files from a specific repository
print("Initializing GitHub Loader...")
loader = GithubFileLoader(
    repo="hwchase17/langchain",
    branch="master",
    relative_path="libs/langchain/langchain/chains",
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: file_path.endswith(".py"),
)

# Note: GithubFileLoader requires the 'pygithub' library
# pip install pygithub

try:
    print("Loading documents from GitHub (this may take a moment)...")
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} documents.")

    if docs:
        print("\n--- Preview of first document ---")
        print(f"Source: {docs[0].metadata.get('source')}")
        print(f"Content: {docs[0].page_content[:200]}...")
except Exception as e:
    print(f"Error: {e}")
    print("Tip: Ensure 'pygithub' is installed (pip install pygithub)")
