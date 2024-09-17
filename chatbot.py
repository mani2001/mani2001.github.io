import os
from langchain.chat_models import ChatGooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"


# Initialize the language model
def initialize_llm():
    llm = ChatGooglePalm(
        model_name="models/chat-bison-001",
        temperature=0,
        max_output_tokens=None,
        top_p=0.95,
        top_k=40,
    )
    return llm


# Load and split the PDF document
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# Optionally calculate chunk IDs
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


# Initialize the embedding function
def get_embedding_function():
    embeddings = GooglePalmEmbeddings()
    return embeddings


# Add chunks to Chroma vector store
def add_to_chroma(chunks, persist_directory, collection_name):
    embedding_function = get_embedding_function()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Check for existing IDs to avoid duplicates
    try:
        existing_items = vectorstore.get()
        existing_ids = set(existing_items["ids"])
    except:
        existing_ids = set()

    new_chunks = []
    for chunk in chunks_with_ids:
        chunk_id = chunk.metadata.get("id")
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        vectorstore.add_documents(new_chunks)
        vectorstore.persist()

    return vectorstore


# Set up the retriever
def setup_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


# Set up the QA chain
def setup_qa_chain(llm, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    return qa_chain


# Function to answer user queries
def answer_query(qa_chain, query):
    result = qa_chain.run(query)
    return result


# Interactive chat function
def chat_with_user(qa_chain):
    print("You can start asking questions (type 'exit' to quit):\n")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = answer_query(qa_chain, user_input)
        print(f"AI: {answer}\n")


# Main execution
if __name__ == "__main__":
    # Initialize the language model
    llm = initialize_llm()

    # Load and process the PDF
    file_path = "static/assets/resume.pdf"
    chunks = load_and_split_pdf(file_path)

    # Add chunks to Chroma vector store
    CHROMA_PATH = "chroma"
    COLLECTION_NAME = "my_collection"
    vectorstore = add_to_chroma(chunks, CHROMA_PATH, COLLECTION_NAME)

    # Set up retriever and QA chain
    retriever = setup_retriever(vectorstore)
    qa_chain = setup_qa_chain(llm, retriever)

    # Start the interactive chat
    chat_with_user(qa_chain)
