from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
output_parser = StrOutputParser()

template1 = PromptTemplate(
    template="""Tell me about the {topic} in detail""",
    input_variables=["topic"]
)

template2 = PromptTemplate(
     template="write a 5 line description for this text.\n{text}",
     input_variables=["text"]
)

chain1 = template1 | model | output_parser
chain2 = template2 | model | output_parser

full_chain = chain1 | chain2

print("--- Result 1 (Full Chain) ---")
result1 = full_chain.invoke({"topic": "blackhole"})
print(result1)

print("\n--- Result 2 (Summary Only) ---")
result2 = chain2.invoke({"text": "A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it."})
print(result2)