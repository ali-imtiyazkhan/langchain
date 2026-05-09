# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the text
with open("example.txt") as f:
    text_data = f.read()

# 2. Initialize the Recursive Splitter
# This splitter is "semantic" because it tries to keep paragraphs and sentences together
# by following a hierarchy of separators: ["\n\n", "\n", " ", ""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# 3. Create chunks
chunks = splitter.create_documents([text_data])

# 4. Results
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)
