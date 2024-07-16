#!/bin/bash

echo "Installing packages..."
pip install --upgrade --quiet langchain-openai tiktoken langchain-chroma langchain GitPython
pip install langchain-community
pip install git
pip install sentence-transformers
pip install torch torchvision torchaudio
pip install tree-sitter==0.21.3
pip install tree-sitter tree-sitter-languages
pip install langchain_huggingface
pip install pymupdf


echo "All packages installed successfully."

