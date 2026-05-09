from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Load the text
with open("example.txt") as f:
    text_data = f.read()

# Initialize the Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Initialize the Semantic Chunker
# breakpoint_threshold_type can be "percentile", "standard_deviation", "interquartile"
splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile"
)

# Create chunks
print("Generating semantic chunks using AI... this may take a moment.")
chunks = splitter.create_documents([text_data])

# Results
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Semantic Chunk {i+1} ---")
    print(chunk.page_content)
