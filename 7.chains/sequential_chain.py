from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

prompt1 = PromptTemplate(
    template = 'generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = '5 pointer of the given {report}',
    input_variables=['report']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic' : "Goat of cricket Rohit Sharma"})

print(result)

chain.get_graph().print_ascii()
