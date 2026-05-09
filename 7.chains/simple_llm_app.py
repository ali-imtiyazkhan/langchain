from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

prompt = PromptTemplate(
    template = 'write a poem on {topic}',
    input_variables=['topic']
)

topic = input("Enter a topic")

formatted_prompt = prompt.format(topic=topic)

response = model.invoke(formatted_prompt)

print(response.content)