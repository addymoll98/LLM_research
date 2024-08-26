# CodeQuery

CodeQuery is a LangChain based RAG LLM program which asks an LLM questions about a codebase. 

## Installation

CodeQuery requires the following packages, which can be installed using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-text-splitters
pip install pysqlite3
pip install llama-cpp-python
pip install chromadb
pip install sentence-transformers
pip install tree-sitter==0.21.3 tree-sitter-languages
pip install langchain_huggingface
```

This program can be run with either llama.cpp or llamafile. 

### Running with llama.cpp
An LLM in a GUFF format must be downloaded locally, and the model_path variable must be updated with the path to this model. The model we have used is zephyr-7b-beta.Q5_K_M.gguf`, which can be dowloaded from [Hugging Face](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF). Update `model` and `model_path` to point to the local LLM GUFF file.

### Running with llamafile
If using llamafile, you need to download a Llamafile model (will have .llamafile as an extention). Before running the program, you need to start up llamafile by running that file as an executable. (For example, on linux  in the terminal: $./mistral-7b-instruct-v0.2.Q4_K_M.llamafile)

Note that llamafile only seems to work with Linux. There is an issue with using llamafile on Windows. 

## Usage

Run the program
```bash
python Annotate.py
```

There are a number of questions included in the code for the LLM to answer. These questions can be modified, or restructured to run as as a loop and allowing the user to type custom questions upon running the program. 
