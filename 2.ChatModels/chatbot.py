from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI()

while True:
    user_input = input("you : ")
    if user_input == "exit":
        break
    response = model.invoke([user_input])
    print("bot : ", response.content)
