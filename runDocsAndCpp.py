import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.llamafile import Llamafile

os.environ['CURL_CA_BUNDLE'] = ''
current_directory = os.getcwd()

# Paths to your repositories
pdf_repo_path = os.path.join(current_directory, "Dafman")
cpp_repo_path = os.path.join(current_directory, "websocketpp")

# Load PDF documents
pdf_loader = DirectoryLoader(pdf_repo_path, loader_cls=PyMuPDFLoader)
pdf_documents = pdf_loader.load()

# Load C++ documents
cpp_loader = GenericLoader.from_filesystem(
    cpp_repo_path,
    glob="**/*",
    suffixes=[".cpp", ".hpp"],
    exclude=[],
    parser=LanguageParser(language=Language.CPP, parser_threshold=500)
)
cpp_documents = cpp_loader.load()

# Combine documents from both sources
all_documents = pdf_documents + cpp_documents

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1)
all_texts = splitter.split_documents(all_documents)

# Embed documents
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
all_embeddings = embeddings.embed_documents([text.page_content for text in all_texts])

if len(all_embeddings) != len(all_texts):
    raise ValueError("Mismatch between the number of embeddings and text chunks.")

# Create Chroma DB
db = Chroma.from_documents(all_texts, embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

### FOR RUNNING WITH HUGGINGFACE ###

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = Insert your token here
# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     model_kwargs={
#         "max_new_tokens": 512,
#         "top_k": 30,
#         "temperature": 0.1,
#         "repetition_penalty": 1.03,
#     },
# )

#####################################

##### FOR RUNNING WITH LLAMAFILE #####

llm = Llamafile()

#####################################

##### FOR RUNNING WITH LLAMACPP #####

# from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain_core.prompts import PromptTemplate    
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# zephr_path = os.path.join(current_directory, "zephyr-7b-beta.Q2_K.gguf")
# llm = LlamaCpp(model_path=zephr_path, verbose=True, n_ctx=2048, callback_manager=callback_manager)

#####################################

# Define prompts and chains
prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's questions based on the below context. Do not ask any follow up questions, only answer the question.:\n\n{context}"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)

#  Chat loop with history
chat_history = []
chat = True

while chat:
    question = input("Enter a question: ")
    chat_history.append({"role": "user", "content": question})
    
    result = qa.invoke({"input": question, "chat_history": chat_history})
    answer = result['answer']
    
    chat_history.append({"role": "assistant", "content": answer})
    
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {answer} \n")
    
    if input("Do you want to ask another question? (Y/N): ").strip().upper() != "Y":
        chat = False

