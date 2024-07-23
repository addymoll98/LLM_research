# Purpose: LangChain Program
# Description:  Retrieval Augmented Generation (RAG) implementation which uses a local LLM with llamacpp-python 
#               to answer questions about a codebase.
# Codebase: AutoRCCar


# Import Required Packages

# Document loading and parsing 

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
#from git import Repo
# For splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# For database and embedding
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.vectorstores import Chroma
import os
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

# For retrieval and chaining support

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.llamafile import Llamafile
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# Loading and Setup

repo_path = "/home/adelinemoll/Public/LLM/LangChain/AutoRCCar"

# Uncomment below if repo is not already installed! The code below clones the RC car repo to the ./repo path
# repo = Repo.clone_from("https://github.com/hamuchiwa/AutoRCCar", to_path=repo_path)

loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
print("Number of Documents loaded: ", len(documents))


######## Splitting ########

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=100
)

texts = python_splitter.split_documents(documents)
print("Number of Texts: ", len(texts))


######## Embedding and Chroma DB storage ######## 
# Texts are stored in the Chroma db with their embeddings using an embedding model (embedding model stored locally)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(
    search_type="mmr",  # Search algorithm. An alternative is "similarity"
    search_kwargs={"k": 8}, # Number of results to retrieve 
)

########  Here we can declare which LLM to use ########
model = "zephyr-7b-beta.Q2_K.gguf" # Change this to specify the llm

# LLM: Llamacpp 
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/home/adelinemoll/Public/LLM/LangChain/LLMs/" + model, 
    verbose=True, 
    n_ctx=4000, 
    callback_manager=callback_manager,
    n_batch=1024,
    n_gpu_layers=-1,
    max_tokens=700
    #stop=["Q:", "\n"]
    )

# LLM: Llamafile (doesn't currently work because of compatibility issues on windows)
# llm = Llamafile()


################################ TEST CODE #################################
# The following block runs runs a series of questions through the llm and outputs the results to a modelname_output.txt file

my_prompt = PromptTemplate.from_template("Use the context below to answer the question. \n Context: {context} \n Question: {input}")

question = "What does the RCControl class do?"
question2 = "What are popular brands of erasers?" # irrelevant question
question3 = "Where is the rc_car.stop() function used?"
question4 = "What is in the rc_driver.py file?"
question5 = "How is video input data handled on the server? Reference the VideoStreamHandler class."
question6 = "How can the object detection mechanism be optimized in the ObjectDetection class?"

question_list = [question]#, question3, question4, question5, question6]
result_list = []

output_file = open("outputs/" + model + "_output.txt", "w")
output_file.write("\n############## RESULTS #############\n")

for index,q in enumerate(question_list):
    retrieved_context = retriever.invoke(q)
    formatted_prompt = my_prompt.format(input=q, context=retrieved_context)

    result = llm.invoke(formatted_prompt)
    result_list.append(result)

    print("\n############## RESULTS #############\n")
    print("Question: ", question_list[index])
    print("\n")
    print("Answer: ", result)
    print("\n\n")

    output_file.write("Question: \n" + question_list[index])
    output_file.write("\n")
    output_file.write("Answer: \n" + result)
    output_file.write("\n\n")

# for index,result in enumerate(result_list):
#     print("Question: ", question_list[index])
#     print("\n")
#     print("Answer: ", result)
#     print("\n\n")

#     output_file.write("Question: \n" + question_list[index])
#     output_file.write("\n")
#     output_file.write("Answer: \n" + result)
#     output_file.write("\n\n")

############################################################################
