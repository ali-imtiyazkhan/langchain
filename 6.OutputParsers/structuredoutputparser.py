# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# # from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from dotenv import load_dotenv
# import os

# load_dotenv()


# response_schemas = [
#     ResponseSchema(name="answer", description="answer to the user's question"),
#     ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
# ]

# parser = StructuredOutputParser.from_response_schemas(response_schemas)


# format_instructions = parser.get_format_instructions()

# # Set up the prompt template
# prompt = PromptTemplate(
#     template="Answer the user's question as best as possible.\n{format_instructions}\n{question}",
#     input_variables=["question"],
#     partial_variables={"format_instructions": format_instructions}
# )

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# # Create the chain
# chain = prompt | model | parser

# # Run the chain
# question = "What is the capital of France?"
# result = chain.invoke({"question": question})

# print(f"Result type: {type(result)}")
# print(result)