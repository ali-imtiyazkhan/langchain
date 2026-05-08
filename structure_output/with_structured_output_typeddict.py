from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Review (TypedDict):
    """schema for the review model"""
    summary : str
    sentiment:str
    rating: int

structure_model = model.with_structured_output(Review)

result = structure_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that cannot be uninstalled. """)

print(result)
