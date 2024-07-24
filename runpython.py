import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.llamafile import Llamafile

# Get the current working directory
current_directory = os.getcwd()

# Construct the full path to your repository
repo_path = os.path.join(current_directory, "AutoRCCar")

# print(f"Repository path: {repo_path}")

# Debug: Check if repo_path exists
if not os.path.exists(repo_path):
    print(f"Repo path does not exist: {repo_path}")
else:
    print(f"Repo path exists: {repo_path}")

# Load documents
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".py"],  # Update to match C++ file extensions
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),  # Update to use CPP language parser
)
documents = loader.load()

# Debug: Print the number of loaded documents
print(f"Loaded {len(documents)} documents.")

# Check if documents are loaded and continue
if not documents:
    print("No documents were loaded. Please check the repository path and file patterns.")
else:
    print(f"Loaded {len(documents)} documents.")

    # Split documents
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=1
    )
    # previous values: chunk_size=2000, chunk_overlap=200
    texts = python_splitter.split_documents(documents)

    # Debug: Print the number of text chunks and first few text chunks
    print(f"Split into {len(texts)} chunks.")
    # print(texts[:5])  # Print first 5 text chunks

    # Embed documents
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Embed all documents
    all_embeddings = embeddings.embed_documents([text.page_content for text in texts])

    # Debug: Verify generated embeddings
    if not all_embeddings:
        print("No embeddings were generated. Check the input documents and the embedding model.")
    else:
        print(f"Generated {len(all_embeddings)} embeddings.")

    # Ensure the list of embeddings matches the text chunks
    if len(all_embeddings) != len(texts):
        raise ValueError("Mismatch between the number of embeddings and text chunks.")

    # Create Chroma DB
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )


    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate


    ### FOR RUNNING WITH HUGGINGFACE ###

    #os.environ["HUGGINGFACEHUB_API_TOKEN"] = Insert your token here
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
    )
    #####################################

    ##### FOR RUNNING WITH LLAMAFILE #####

    # llm = Llamafile()

    #####################################

    ##### FOR RUNNING WITH LLAMACPP #####

    # from langchain_community.llms import LlamaCpp
    # from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
    # from langchain_core.prompts import PromptTemplate    
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # zephr_path = os.path.join(current_directory, "zephyr-7b-beta.Q2_K.gguf")
    # llm = LlamaCpp(model_path=zephr_path, verbose=True, n_ctx=2048, callback_manager=callback_manager)

    #####################################

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
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    qa = create_retrieval_chain(retriever_chain, document_chain)


    chat = True
    while chat:
        print("Enter a question: ")
        question = input()
        result = qa.invoke({"input": question})
        print(len(result))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
        print("Do you want to ask another question? (Y/N)")
        if (input().strip().upper() != "Y"):
            chat = False
    
    
    # questions = [
    #     "What does the class spin_mutex do?",
    #     "Explain what this cppcoro program is",
    # ]

    # for question in questions:
    #     result = qa.invoke({"input": question})
    #     print(len(result))
    #     print(f"-> **Question**: {question} \n")
    #     print(f"**Answer**: {result['answer']} \n")
