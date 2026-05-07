from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


embed_model=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

embedding = embed_model.embed_query("Hello, how are you?")

print(len(embedding))