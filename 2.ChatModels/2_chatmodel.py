from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

prompt=[
    {"role": "user", "content": "Hello, what is the captial of india?"}
]
llm = ChatAnthropic(model="claude-3.5-sonnet-20240620", temperature=0)

response = llm.invoke(prompt)

print(response.content)