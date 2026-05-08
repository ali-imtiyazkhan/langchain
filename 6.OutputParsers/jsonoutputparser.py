from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Joke(BaseModel):
    setup: str = Field(description="the setup of the joke")
    punchline: str = Field(description="the punchline of the joke")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Tell me a joke about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

topic = "ice cream"
result = chain.invoke({"topic": topic})

print(f"Data Type: {type(result)}")
print(result)
