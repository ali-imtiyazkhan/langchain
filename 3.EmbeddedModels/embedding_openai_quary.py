from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embedding = embed_model.embed_query("Hello, how are you?")

print(len(embedding))