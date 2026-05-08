from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = []

while True:
    user_input = input("you : ")
    chat_history.append(user_input)
    if user_input == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(response.content)
    print("bot : ", response.content)

print(chat_history)
