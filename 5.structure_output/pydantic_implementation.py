from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review (BaseModel):
    summary : str = Field(description="summary of the review (2-3 lines)")
    sentiment:str = Field(description="sentiment of the review (positive, negative, neutral)")
    rating: int = Field(description="rating of the review (1-5)")

structure_model = model.with_structured_output(Review)

result = structure_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that cannot be uninstalled. """)

print(result.model_dump_json(indent=2))

