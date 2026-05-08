# LangChain Learning Journey 🦜🔗

This repository is dedicated to my journey of learning **LangChain** in Python. It contains various examples, experiments, and implementations of LangChain components, from basic LLM calls to complex structured outputs.

## 📁 Project Structure

The project is organized into chronological modules:

- **`1.LLMs/`**: Introduction to Large Language Models (LLMs) and basic invocation.
- **`2.ChatModels/`**: Working with Chat-specific models including OpenAI, Google Gemini, and Hugging Face.
- **`3.EmbeddedModels/`**: Implementing text embeddings for semantic search and document comparison.
- **`4.Messages/`**: Understanding and managing different message types: `SystemMessage`, `HumanMessage`, and `AIMessage`.
- **`5.structure_output/`**: Techniques for getting structured data from LLMs using `TypedDict` and `Pydantic`.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Virtual Environment (recommended)

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

## 🛠️ Key Technologies
- **LangChain**: Framework for building LLM-powered applications.
- **Google Gemini**: Using `langchain-google-genai`.
- **OpenAI**: Using `langchain-openai`.
- **Pydantic**: For data validation and structured output.
- **Dotenv**: For managing environment variables.

---
*Happy Coding!* 🚀
