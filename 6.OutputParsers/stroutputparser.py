from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1st prompt 
template1 = PromptTemplate(
    template="""Tell me about the {topic} in detail""",
    input_variables=["topic"]
)

# 2nd prompt 
template2 = PromptTemplate(
     template="write a 5 line description for this text.\n{text}",
     input_variables=["text"]
)

# Formatting the prompts
prompt1 = template1.format(topic="Python")
prompt2 = template2.format(text="The quick brown fox jumps over the lazy dog. This is a classic pangram used to test typefaces and keyboards.")

# Invoking Gemini
result1 = model.invoke(prompt1)
print("--- Result 1 ---")
print(result1.content)

print("\n--- Result 2 ---")
result2 = model.invoke(prompt2)
print(result2.content)