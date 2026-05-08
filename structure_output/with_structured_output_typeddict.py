from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Review (TypedDict):
    """schema for the review model"""
    summary : Annotated[str, "summary of the review (2-3 lines)"]
    sentiment:Annotated[str, "sentiment of the review (positive, negative, neutral)"]
    rating: Annotated[int, "rating of the review (1-5)"]

structure_model = model.with_structured_output(Review)

result = structure_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that cannot be uninstalled. """)

print(result)
