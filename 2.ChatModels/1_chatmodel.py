from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

prompt=[
    {"role": "user", "content": "Hello, what is the captial of india?"}
]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

response = llm.invoke(prompt)

print(response.content)