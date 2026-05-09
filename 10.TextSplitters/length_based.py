# pyrefly: ignore [missing-import]
from langchain_text_splitters import CharacterTextSplitter

with open("example.txt") as f:
    text_data = f.read()

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# 3. Create the chunks
chunks = splitter.create_documents([text_data])

# 4. Print the results
print(f"Original text length: {len(text_data)}")
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Length: {len(chunk.page_content)}")
    print(f"Content: {chunk.page_content}")
