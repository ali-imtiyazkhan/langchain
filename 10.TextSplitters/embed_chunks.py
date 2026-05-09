from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 1. Prepare chunks
with open("example.txt") as f:
    text_data = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(text_data)

print(f"Number of chunks: {len(chunks)}")

# 2. Initialize the Embedding Model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. Convert Chunks to Embeddings
print("Converting chunks to embeddings...")
vectors = embeddings_model.embed_documents(chunks)

# 4. Show the results
for i, vector in enumerate(vectors):
    print(f"\n--- Embedding for Chunk {i+1} ---")
    print(f"Text: {chunks[i]}")
    print(f"Vector (first 5 dimensions): {vector[:5]}...")
    print(f"Total Dimensions: {len(vector)}")
