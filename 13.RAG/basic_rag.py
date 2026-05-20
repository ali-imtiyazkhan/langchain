from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

#Load the existing FAISS vector store
print("Loading vector store from 'faiss_index'...")
try:
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"Error loading vector store: {e}")
    exit()

# Define the RAG prompt template
template = """Answer the question based ONLY on the following context:
{context}

Question: {question}

Helpful Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Helper function to format documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Create the RAG chain using LCEL (LangChain Expression Language)
# - context: retrieves relevant documents and formats them
# - question: passes the user's input directly through
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 6. Execute the RAG chain
query = "What is LangChain and what are its core components?"
print(f"\nUser Query: {query}")
print("Thinking...")

try:
    response = rag_chain.invoke(query)
    print("\n--- AI Response (using retrieved context) ---")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
