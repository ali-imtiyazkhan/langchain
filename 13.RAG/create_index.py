import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

#Load environment variables
load_dotenv()

def create_index():
    print("1. Loading document...")
    loader = TextLoader("sample_data.txt")
    documents = loader.load()

    print("2. Splitting document...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("3. Creating embeddings and vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vector_store = FAISS.from_documents(docs, embeddings)

    print("4. Saving vector store locally...")
    vector_store.save_local("faiss_index")
    print("Done! Vector store saved to 'faiss_index' directory.")

if __name__ == "__main__":
    create_index()
