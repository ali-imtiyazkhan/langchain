from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("you : ")
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    print("bot : ", response.content)
    chat_history.append(AIMessage(content=response.content))

print(chat_history)