import os
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-FY0a-yMJXduW-KncGHddMIOgQ-dDsfc"


# Function to initialize the language model
def initialize_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=1000,
        max_retries=2,
    )
    return llm


# Function to load and split the PDF document
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


# Function to initialize the embedding function
def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Test if the embeddings function works
    return embeddings


# Function to add chunks to Chroma vector store
def add_to_chroma(chunks, persist_directory, collection_name):
    embedding_function = get_embedding_function()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Add documents to the vector store
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return vectorstore


# Function to set up the retriever
def setup_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


# Function to set up the QA chain
def setup_qa_chain(llm, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    return qa_chain


# Initialize the QA chain when the app starts
llm = initialize_llm()
file_path = "static/assets/resume.pdf"
chunks = load_and_split_pdf(file_path)
CHROMA_PATH = "chroma"
COLLECTION_NAME = "my_collection"
vectorstore = add_to_chroma(chunks, CHROMA_PATH, COLLECTION_NAME)
retriever = setup_retriever(vectorstore)
qa_chain = setup_qa_chain(llm, retriever)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chatbot/", methods=["POST"])
def chatbot():
    data = request.get_json()
    message = data.get("message", "")
    # Use the qa_chain to get the response
    response = qa_chain.run(message)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
