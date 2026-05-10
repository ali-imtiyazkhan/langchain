from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize the Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create LangChain documents for IPL players
print("Creating documents...")
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]

# Initialize Chroma Vector Store
print("Initializing Chroma...")
persist_directory = 'my_chroma_db_ipl'
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory,
    collection_name='ipl_players'
)

# Add documents and get IDs
print("Adding documents...")
ids = vector_store.add_documents(docs)
print(f"Added documents with IDs: {ids}")

# View documents
print("\n--- Viewing all documents in collection ---")
data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print(f"Total documents: {len(data['ids'])}")

# Search documents
print("\n--- Similarity Search: 'Who among these are a bowler?' ---")
results = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
for doc in results:
    print(f"Content: {doc.page_content} | Metadata: {doc.metadata}")

# Search with similarity score
print("\n--- Similarity Search with Score ---")
results_with_score = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
for doc, score in results_with_score:
    print(f"Score: {score:.4f} | Content: {doc.page_content}")

# Meta-data filtering
print("\n--- Filtering by team: 'Chennai Super Kings' ---")
filtered_results = vector_store.similarity_search_with_score(
    query="all rounder",
    filter={"team": "Chennai Super Kings"}
)
for doc, score in filtered_results:
    print(f"Score: {score:.4f} | Content: {doc.page_content}")

# Update documents
print("\n--- Updating document for Virat Kohli ---")
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# Using the first ID (which corresponds to Kohli in this run)
vector_store.update_document(document_id=ids[0], document=updated_doc1)
print("Document updated.")

# Delete document
print(f"\n--- Deleting document with ID: {ids[0]} ---")
vector_store.delete(ids=[ids[0]])
print("Document deleted.")

# Final check
final_data = vector_store.get()
print(f"\nFinal document count: {len(final_data['ids'])}")
