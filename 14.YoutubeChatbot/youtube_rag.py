import os
import sys
import urllib.parse as urlparse
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

def extract_video_id(url):
    """Extracts the 11-character video ID from a YouTube URL."""
    url_data = urlparse.urlparse(url)
    query = urlparse.parse_qs(url_data.query)
    video_id = query.get("v")
    if video_id:
        return video_id[0]
    path = url_data.path.split('/')
    if path:
        return path[-1]
    return None

def load_youtube_transcript(url):
    """Fetches the transcript for a YouTube video robustly, translating it to English if needed."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract a valid Video ID from the provided URL.")
        
    # Get the list of available transcripts
    transcript_list = YouTubeTranscriptApi().list(video_id)
    
    try:
        #Try to get an English transcript (either manual or auto-generated)
        transcript = transcript_list.find_transcript(['en'])
        language_info = "English"
    except Exception:
        # Fall back to the first available transcript (e.g., Hindi) and translate it to English
        first_transcript = next(iter(transcript_list))
        language_info = f"{first_transcript.language} (translated to English)"
        if first_transcript.is_translatable:
            transcript = first_transcript.translate('en')
        else:
            transcript = first_transcript
            language_info = f"{first_transcript.language} (no translation available)"
            
    # Fetch the actual transcript entries
    data = transcript.fetch()
    
    # Concatenate transcript text safely
    texts = []
    for entry in data:
        if isinstance(entry, dict):
            texts.append(entry.get('text', ''))
        else:
            texts.append(getattr(entry, 'text', ''))
    full_text = " ".join(texts)
    
    # Return as a LangChain Document
    return [Document(
        page_content=full_text, 
        metadata={"source": video_id, "language": transcript.language_code, "language_info": language_info}
    )]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print(" YouTube Video RAG Chatbot  ")
    
    # Get YouTube URL from user
    url = input("\nEnter YouTube Video URL: ").strip()
    if not url:
        print("Error: YouTube URL cannot be empty.")
        return

    try:
        # Load the transcript using our robust loader
        print("\n[1/4] Fetching video transcript...")
        documents = load_youtube_transcript(url)
        
        doc = documents[0]
        print(f"Successfully loaded transcript!")
        print(f" Video ID: {doc.metadata['source']}")
        print(f" Language: {doc.metadata['language_info']}")

        # Split transcript into chunks
        print("\n[2/4] Splitting transcript into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        print(f"Split transcript into {len(docs)} chunks.")

        # Embed chunks and save in Vector Store (in-memory FAISS)
        print("\n[3/4] Creating embeddings and building vector index...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        print("Vector store initialized successfully!")

        # Set up the Prompt, LLM, and RAG Chain
        print("\n[4/4] Setting up Chatbot chain...")
        model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

        template = """You are a helpful assistant answering questions about a YouTube video based on its transcript.
Use ONLY the following context retrieved from the video transcript to answer the question. If you don't know the answer or if the context doesn't contain the answer, say "I cannot find the answer in the video transcript." Do not make up information.

Context:
{context}

Question: {question}

Helpful Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Build LCEL chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        
        print("\nChatbot is ready! Ask your questions about the video.")
        print("Type 'exit' or 'quit' to end the chat.")
        print("-" * 50)

        # 6. Interactive Chat Loop
        while True:
            try:
                query = input("\nYou: ").strip()
                if not query:
                    continue
                if query.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                
                print("Thinking...")
                response = rag_chain.invoke(query)
                print(f"\nAI: {response}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")

    except Exception as e:
        print(f"\nFailed to load YouTube transcript. Error details: {e}")
        print("Ensure the video has closed captions/subtitles enabled.")

if __name__ == "__main__":
    main()
