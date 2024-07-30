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

os.environ["HUGGINGFACEHUB_API_TOKEN"] = # Inert API Token Here
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 1024,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

model = "huggingface-zephyr-7b-beta-withDAFMANs-Q2" # Change this to specify the llm

# # LLM: Llamacpp 
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# llm = LlamaCpp(
#     model_path="/home/adelinemoll/Public/LLM/LangChain/LLMs/" + model, 
#     verbose=True, 
#     n_ctx=4000, 
#     callback_manager=callback_manager,
#     n_batch=1024,
#     n_gpu_layers=-1,
#     max_tokens=700
#     #stop=["Q:", "\n"]
#     )

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
8.4. Self-Modifying Code. The software shall not have the ability to modify its own instructions
or the instructions of any other application. (T-1). Verification activities should prove that the
certified configuration is unable to modify its own instructions or the instructions of other
applications. A recommended way of partially meeting this requirement is using memory
protections as provided in paragraph 9.3 and paragraph 10.3.
8.5. Program Loading and Initialization.
8.5.1. The software shall execute only after all program instructions, programming files, and
data are loaded and verified. (T-1). Verification activities should prove that software only
executes after all loading and verification are complete.
8.5.2. The software shall communicate results of the program load verification to the system
operators or the crew. (T-1). Verification activities should prove that software communicates
the results of the program load verification described in paragraph 8.5.1 to the system operator
or the crew, or to external systems with the intent of communicating the results to the system
operator or the crew.
8.5.3. The system shall not assume programs have correctly loaded until receiving an
affirmative load status. (T-1). Verification activities should prove that the system treats failure
as the default load status.
8.5.4. The software shall perform volatile memory initialization prior to the execution of the
main application. (T-1). Verification activities should prove that software performs volatile
memory initialization by writing all zeros or a known pattern into memory prior to the
execution of the main application.
8.5.5. The software shall load all non-volatile memory with executable code, data, or a non-
use pattern that the weapon system detects and processes safely upon execution. (T-1).
Verification activities should prove that software loads all non-volatile memory with known
data; non-use patterns cause the processor to respond in a known manner.
8.6. Memory Protection.
8.6.1. The system shall provide at a minimum hardware double bit error detection and single
bit correction on all volatile memory. (T-1). Verification activities should prove that hardware
provides double bit error detection and single bit correction on all volatile memory.
8.6.2. For memory protection that is software-enabled, the software shall enable at a minimum
double bit error detection and single bit correction on all volatile memory. (T-1). Verification
activities should prove that software enables at a minimum double bit error detection and single
bit correction when not automatically enabled by hardware.
8.7. Declassification and Zeroize Functionality. The software shall provide methods to erase
or obliterate, as appropriate for the memory technology, any unencrypted classified or controlled32
AFMAN91-119 11 MARCH 2020
information from memory using National Security Agency-approved design criteria found in DoD
Instruction (DoDI) S-5200.16, Objectives and Minimum Standards for Communications Security
(COMSEC) Measures Used in Nuclear Command and Control (NC2) Communications (U). (T-
1). Verification activities should prove that software provides methods to erase or obliterate any
clear-text secure codes.
Question: Does the given code about a self driving rc car given as context comply with the requirements above?"""

question_list = [question7]
result_list = []

output_file = open("huggingface_outputs/" + model + "_output.txt", "w")
output_file.write("Setup: \n Using a huggingface API model to answer questions about the DAFMAN requirements relating to the rc car codebase")
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
