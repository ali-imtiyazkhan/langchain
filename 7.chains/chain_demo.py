from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(template=prompt_template, input_variables=["product"])

parser = StrOutputParser()

lcel_chain = prompt | model | parser

print("\n--- Modern LCEL Output ---")
lcel_result = lcel_chain.invoke({"product": "eco-friendly water bottles"})
print(lcel_result)

print("\n--- Batch Mode (Parallel) ---")
results = lcel_chain.batch([{"product": "shoes"}, {"product": "watches"}])
for result in results:
    print(result)

print("\n--- Streaming Output (Real-time) ---")
for chunk in lcel_chain.stream({"product": "solar panels"}):
    print(chunk, end="|", flush=True)

lcel_chain.get_graph().print_ascii()
