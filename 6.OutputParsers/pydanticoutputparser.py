from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chain = prompt | model | parser


query = "Tell me a joke about bears."
result = chain.invoke({"query": query})

print(f"Data type: {type(result)}")
print(f"Setup: {result.setup}")
print(f"Punchline: {result.punchline}")
