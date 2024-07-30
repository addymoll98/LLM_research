# Purpose: LangChain Program
# Description:  Retrieval Augmented Generation (RAG) implementation which uses a LLM from the huggingface API
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
    search_kwargs={"k": 20}, # Number of results to retrieve 
)

########  Here we can declare which LLM to use ########

## FOR RUNNING WITH HUGGINGFACE ###

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     model_kwargs={
#         "max_new_tokens": 1024,
#         "top_k": 30,
#         "temperature": 0.1,
#         "repetition_penalty": 1.03,
#     },
# )

model = "zephyr-7b-beta.Q5_K_M.gguf" # Change this to specify the llm
# Options are: zephyr-7b-beta.Q5_K_M.gguf, deepseek-coder-v2-lite-instruct-q4_k_m.gguf, zephyr-7b-beta.Q2_K.gguf

# LLM: Llamacpp 
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/home/adelinemoll/Public/LLM/LangChain/LLMs/" + model, 
    verbose=True, 
    n_ctx=50000, 
    callback_manager=callback_manager,
    n_batch=1024,
    n_gpu_layers=-1,
    max_tokens=8000
    #stop=["Q:", "\n"]
    )

# LLM: Llamafile (doesn't currently work because of compatibility issues on windows)
# llm = Llamafile()


################################ TEST CODE #################################
# The following block runs runs a series of questions through the llm and outputs the results to a modelname_output.txt file

my_prompt = PromptTemplate.from_template("Instructions: You will be asked a question on a codebase. Use the context about the codebase below to answer the question. Only answer questions relevant to the codebase. If you don't know the answer, simply say so. Do not make up code that doesn't exist in the codebase.\n Context: {context} \n Question: {input}")

question = "What does the RCControl class do?"
question2 = "What are popular brands of erasers?" # irrelevant question
question3 = "Where is the rc_car.stop() function used?"
question4 = "What is in the rc_driver.py file?"
question5 = "How is video input data handled on the server? Reference the VideoStreamHandler class."
question6 = "How can the object detection mechanism be optimized in the ObjectDetection class?"
question7 = """ Consider the following requirements:
6.2. Transmission of Critical Software or Critical Data.
6.2.1. The transmission of Critical Software or Critical Data outside of immediate control of
the weapon system can become a safety concern if the data is susceptible to intentional or
accidental manipulation.
6.2.2. The software shall use protocols that protect the transmission of Critical Software via
over-the-air broadcasts or transmission over media outside of immediate control of the weapon
system from inadvertent or intentional corruption, through encryption, digital signatures, or
similar methods. (T-1). Verification activities should prove that protocol protection
mechanisms protect Critical Software during transmission of over-the-air broadcasts or
transmission over media outside of immediate control of the weapon system. If the weapon
system stores the Critical Software in an incorruptible manner, and the weapon system verifies
the Critical Software during each restart, then this requirement no longer applies. Encryption
is the preferred mechanism for protocol protection, but the National Security Agency should
approve the encryption methodology.
Question: Given the context about the codebase and the requirements above, 
Does the given code about a self driving rc car given as context comply with the requirements above?"""

question_list = [question7]
result_list = []

output_file = open("llamacpp_outputs/" + model + "DAFMAN-Q1" + "_output.txt", "w")
output_file.write("Setup: \n Using a local llm to answer questions about the DAFMAN requirements relating to the rc car codebase")
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

############################################################################
