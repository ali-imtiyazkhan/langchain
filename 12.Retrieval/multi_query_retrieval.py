from dotenv import load_dotenv
import logging
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# Set up logging to see the generated queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load the existing vector store
print("Loading vector store (FAISS)...")
vector_store = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Initialize the LLM for query generation
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Initialize the MultiQueryRetriever
# It generates multiple queries from the user input to improve recall
print("Initializing MultiQuery Retriever...")
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm
)

# Perform retrieval
query = "What can you tell me about the architecture of LangChain?"
print(f"\nRetrieving documents for: '{query}'...")

try:
    # This will trigger the LLM to generate multiple queries
    docs = retriever.invoke(query)

    print(f"\n--- Retrieved {len(docs)} unique documents ---")

    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content[:200]}...") # Showing snippet
        print(f"Metadata: {doc.metadata}")

except Exception as e:
    print(f"Error: {e}")
