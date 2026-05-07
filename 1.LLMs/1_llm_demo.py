from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

prompt=[
    {"role": "user", "content": "Hello, what is the captial of india?"}
]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

response = llm.invoke(prompt)

print(response.content)