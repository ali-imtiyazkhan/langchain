from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-flash-latest")

model2 = ChatGoogleGenerativeAI(model="gemini-flash-latest")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'generate short and simple notes on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'generate 5 questions from the given notes {notes}',
    input_variables=['notes']
)


prompt3 = PromptTemplate(
    template = 'merge the privided notes and questions {notes} and give me the best {questions} in single document',
    input_variables=['notes','questions']
)

# Fix: Use RunnableParallel to pass both 'notes' and 'questions' to prompt3
chain = (
    prompt1 
    | model1 
    | parser 
    | RunnableParallel(notes=RunnablePassthrough(), questions=prompt2 | model1 | parser) 
    | prompt3 
    | model2 
    | parser
)

result = chain.invoke({'topic': 'Generative AI'})
print(result)

chain.get_graph().print_ascii() 