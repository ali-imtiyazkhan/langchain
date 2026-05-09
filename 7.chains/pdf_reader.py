from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

loader = PyPDFLoader("sample.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

# 6. Define Prompt Template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# 7. Create the RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 8. Run the chain (Example)
if __name__ == "__main__":
    question = "What is the main topic of this PDF?"
    try:
        result = rag_chain.invoke(question)
        print("\n--- Response ---")
        print(result)
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Make sure 'sample.pdf' exists and your GOOGLE_API_KEY is set in .env")
