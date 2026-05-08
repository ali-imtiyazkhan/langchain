from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

schema_path = os.path.join(os.path.dirname(__file__), "json_schema.json")
with open(schema_path, "r") as f:
    student_schema = json.load(f)

structured_model = model.with_structured_output(student_schema)

result = structured_model.invoke("My name is Ali, I am 23 years old and my roll number is 101.")

# print the result in json format with proper indentation
print(json.dumps(result, indent=2))
