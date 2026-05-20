import os
import sys
import re
import getpass
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# ANSI colours for pretty CLI
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GREY   = "\033[90m"
    BLUE   = "\033[94m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════╗
║       GitHub Codebase Chatbot  •  RAG Edition        ║
║     Powered by Gemini Flash + Google Embeddings      ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")

# Extension → LangChain Language mapping
EXTENSION_LANGUAGE_MAP = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".jsx":   Language.JS,
    ".ts":    Language.JS,
    ".tsx":   Language.JS,
    ".java":  Language.JAVA,
    ".cpp":   Language.CPP,
    ".cc":    Language.CPP,
    ".cxx":   Language.CPP,
    ".c":     Language.C,
    ".go":    Language.GO,
    ".rb":    Language.RUBY,
    ".rs":    Language.RUST,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".kt":    Language.KOTLIN,
    ".md":    Language.MARKDOWN,
    ".html":  Language.HTML,
    ".sol":   Language.SOL,
}

# Files / directories to skip during loading
EXCLUDED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".exe", ".bin", ".whl",
    ".lock", ".sum", ".mod",
}
EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "coverage",
}


# 1. Credential helpers
def get_env_or_prompt(env_key, prompt_text, secret=False):
    """Return env var value, or prompt the user if it is missing."""
    value = os.getenv(env_key, "").strip()
    if value:
        return value
    if secret:
        value = getpass.getpass(f"{C.YELLOW}{prompt_text}: {C.RESET}").strip()
    else:
        value = input(f"{C.YELLOW}{prompt_text}: {C.RESET}").strip()
    if not value:
        print(f"{C.RED}Error: {env_key} is required.{C.RESET}")
        sys.exit(1)
    return value


# NEW: normalize full GitHub URLs to owner/repo
def normalize_repo_input(raw):
    raw = raw.strip().rstrip("/")
    # Match https://github.com/owner/repo (with optional .git or sub-paths)
    m = re.match(r"(?:https?://)?github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?(?:/.*)?$", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # Already owner/repo format
    if re.match(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$", raw):
        return raw
    return None


# 2. GitHub document loader
def should_include(file_path):
    """Return True if the file should be indexed."""
    p = Path(file_path)
    for part in p.parts:
        if part in EXCLUDED_DIRS:
            return False
    if p.suffix.lower() in EXCLUDED_EXTENSIONS:
        return False
    return True


def load_github_documents(repo, branch, token):
    """Load all indexable files from a GitHub repository via API."""
    print(f"\n{C.CYAN}[1/4] Connecting to GitHub repo: {C.BOLD}{repo}{C.RESET} (branch: {branch})")

    loader = GithubFileLoader(
        repo=repo,
        branch=branch,
        access_token=token,
        github_api_url="https://api.github.com",
        file_filter=should_include,
    )

    print(f"{C.GREY}   Fetching files... this may take a moment for large repos.{C.RESET}")
    docs = loader.load()
    print(f"{C.GREEN}   Loaded {len(docs)} files from GitHub.{C.RESET}")
    return docs


# 3. Language-aware splitting
def split_documents(docs):
    """Split each document using the appropriate language-aware splitter."""
    print(f"\n{C.CYAN}[2/4] Splitting documents into chunks...{C.RESET}")
    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
    )

    all_chunks = []
    language_stats = {}

    for doc in docs:
        ext = Path(doc.metadata.get("source", "")).suffix.lower()
        lang = EXTENSION_LANGUAGE_MAP.get(ext)

        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1500, chunk_overlap=200
            )
            label = lang.value
        else:
            splitter = default_splitter
            label = "generic"

        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
        language_stats[label] = language_stats.get(label, 0) + len(chunks)

    print(f"{C.GREEN}   Split into {len(all_chunks)} chunks.{C.RESET}")
    print(f"{C.GREY}   Language breakdown: {dict(sorted(language_stats.items()))}{C.RESET}")
    return all_chunks


# 4. Vector store
def build_vector_store(chunks):
    """Embed chunks and build an in-memory FAISS index."""
    print(f"\n{C.CYAN}[3/4] Generating embeddings and building vector index...{C.RESET}")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Process in batches to avoid rate-limit issues
    BATCH = 100
    db = None

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        print(
            f"{C.GREY}   Embedding batch {i // BATCH + 1}/{(len(chunks) + BATCH - 1) // BATCH} "
            f"({len(batch)} chunks)...{C.RESET}",
            end="\r",
        )
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    print(f"\n{C.GREEN}   FAISS index built with {len(chunks)} vectors.{C.RESET}")
    return db


# 5. Helper function to format retrieved documents for the prompt
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"--- File: {source} ---\n{doc.page_content}")
    return "\n\n".join(formatted)


# 6. Interactive chat loop
def chat_loop(rag_chain, repo):
    print(f"""
{C.CYAN}{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Chat session started for {repo}
  Type 'exit' or 'quit' to end the chat.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}
""")

    while True:
        try:
            user_input = input(f"{C.GREEN}You  > {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.YELLOW}Session ended.{C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print(f"{C.YELLOW}Goodbye!{C.RESET}")
            break

        print(f"{C.GREY}Thinking...{C.RESET}")
        try:
            response = rag_chain.invoke(user_input)
            print(f"\n{C.BLUE}{C.BOLD}Bot  > {C.RESET}{response}\n")
        except Exception as exc:
            print(f"{C.RED}Error: {exc}{C.RESET}\n")

        print(f"{C.GREY}{'─' * 60}{C.RESET}")


# 7. Entry point
def main():
    banner()

    github_token = get_env_or_prompt(
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "GitHub Personal Access Token (PAT)",
        secret=True,
    )
    google_api_key = get_env_or_prompt(
        "GOOGLE_API_KEY",
        "Google API Key (for Gemini + Embeddings)",
        secret=True,
    )

    # Repository details — accepts full URL or owner/repo
    repo_raw = input(
        f"{C.YELLOW}Repository (e.g. owner/repo-name or full GitHub URL): {C.RESET}"
    ).strip()
    repo = normalize_repo_input(repo_raw)
    if not repo:
        print(f"{C.RED}Invalid repo. Use owner/repo-name or https://github.com/owner/repo.{C.RESET}")
        sys.exit(1)
    print(f"{C.GREY}   Using repo: {repo}{C.RESET}")

    branch = input(
        f"{C.YELLOW}Branch [main]: {C.RESET}"
    ).strip() or "main"

    # Pipeline
    docs = load_github_documents(repo, branch, github_token)
    if not docs:
        print(f"{C.RED}No documents loaded. Check repo name, branch, and token permissions.{C.RESET}")
        sys.exit(1)

    chunks = split_documents(docs)
    db = build_vector_store(chunks)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

    # Set up the Prompt, LLM, and RAG Chain (LCEL)
    print(f"\n{C.CYAN}[4/4] Setting up Chatbot chain...{C.RESET}")
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

    template = """You are a helpful assistant that answers questions about a GitHub codebase.
Use ONLY the following code context retrieved from the repository to answer the question.
If you don't know the answer or the context doesn't contain enough information, say "I cannot find the answer in the codebase."
Do not make up information. When referencing code, mention the file name.

Context:
{context}

Question: {question}

Helpful Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Build LCEL chain (same pattern as YouTube chatbot)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    print(f"{C.GREEN}   Chatbot chain ready!{C.RESET}")

    # Chat
    chat_loop(rag_chain, repo)


if __name__ == "__main__":
    main()