# Retrieval-Augmented Generation (RAG)

This module covers the concepts and implementation of Retrieval-Augmented Generation (RAG) using LangChain. 

## RAG Architecture

The following flowchart represents the standard RAG pipeline, which is divided into an **Ingestion** phase and a **Retrieval & Generation** phase.

```mermaid
graph TD
    %% Ingestion Phase
    WWW[Document Loader] --> Doc[Document]
    Doc --> TS[Text Splitter]
    
    TS --> C1[Chunk 1]
    TS --> C2[Chunk 2]
    TS --> C3[Chunk 3]
    
    C1 --> EM[Embedding Model]
    C2 --> EM
    C3 --> EM
    
    EM --> V1[Vector]
    EM --> V2[Vector]
    EM --> V3[Vector]
    
    V1 --> VS[(Vector Store)]
    V2 --> VS
    V3 --> VS
    
    %% Retrieval & Generation Phase
    Q1((Query)) --> Ret[Retriever]
    Ret -- "Semantic Search" --> VS
    VS -- "Most Relevant Chunks (Context)" --> Ctx[Context]
    
    Q2((Query)) --> Ctx
    
    Ctx --> P[Prompt]
    P --> L[LLM]
    L --> R[Response]

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef blue fill:#bbdefb,stroke:#1976d2,stroke-width:1px;
    classDef green fill:#c8e6c9,stroke:#388e3c,stroke-width:1px;
    classDef yellow fill:#fff9c4,stroke:#fbc02d,stroke-width:1px;
    classDef pink fill:#ffcdd2,stroke:#d32f2f,stroke-width:1px;
    
    class EM,P blue;
    class L green;
    class Ctx,WWW,Doc,TS,C1,C2,C3 yellow;
    class Ret pink;
```

### Components:
1. **Document Loader**: Loads data from a source (e.g., website, file).
2. **Text Splitter**: Breaks the large document into smaller, manageable chunks.
3. **Embedding Model**: Converts the text chunks into numerical vectors.
4. **Vector Store**: A database to store and quickly search these vectors.
5. **Retriever**: Takes the user's query and performs a semantic search against the Vector Store.
6. **Prompt**: Combines the retrieved context with the user's original query.
7. **LLM**: The Large Language Model processes the combined prompt to generate a final response.
