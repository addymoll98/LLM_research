LangChain Basics: 
A framework for making programs that use Large Language Models (LLMs)
 - integrates with many third party libraries
 - contains many tools (doc manipulation, embedding, databases, retriever chains) 
 that are useful for Natural Language Processing (NLP)


Getting Started:
- Open terminal
- Run "setup.sh" to download all packages if not already downloaded
    - If you are using Llamafile(), you need to initialize Llamafile
        - Run ./mistral-7b-instruct-v0.2.Q4_K_M.llamafile --server
- Run "python {filename}" to run the program
- Note: API tokens and paths to repo and llm might need to be updated

How it works: 
- The program loads the codebase into the "repo" folder from git if not already loaded
- A loader is called to get all the documents of the codebase
- Those loaded documents are split into texts
- The texts are stored in a database as embeddings and documents using an embedding function
- A retrieval chain is created that specifies how the LLM will answer the prompt using a retriever and a document chain 
- The retrieval chain is invoked with a prompt and the output is displayed

Switching between Models:
- To swich between HuggingFace API, llamacpp, and llamafile, toggle the comments in lines 105-136

