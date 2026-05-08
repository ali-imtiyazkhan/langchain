from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-flash-latest")
parser = StrOutputParser()

class FeedbackType(BaseModel):
    feedback: str = Field(description="The feedback")
    feedback_type: Literal["positive", "negative"] = Field(description="The type of feedback")

pydantic_parser = PydanticOutputParser(pydantic_object=FeedbackType)

prompt1 = PromptTemplate(
    template="classify the feedback as positive or negative.\n{format_instructions}\nFeedback: {feedback}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

positive_prompt = PromptTemplate(
    template="The user had a positive experience! Write a short thank you note for their feedback: {feedback}",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="The user had a negative experience. Write a polite apology and ask how we can improve based on: {feedback}",
    input_variables=["feedback"]
)

classification_chain = prompt1 | model1 | pydantic_parser

positive_chain = positive_prompt | model1 | parser
negative_chain = negative_prompt | model1 | parser

branch = RunnableBranch(
    (lambda x: x.feedback_type == "positive", (lambda x: {"feedback": x.feedback}) | positive_chain),
    (lambda x: x.feedback_type == "negative", (lambda x: {"feedback": x.feedback}) | negative_chain),
    parser
)

full_chain = classification_chain | branch

print("--- Positive Feedback ---")
print(full_chain.invoke({"feedback": "I love this product, it works perfectly!"}))

print("\n--- Negative Feedback ---")
print(full_chain.invoke({"feedback": "This is terrible, it broke after one day."}))
