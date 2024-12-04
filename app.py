import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import tempfile

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to handle document embeddings
def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:

        # Initialize the embeddings
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load PDFs from uploaded files
        documents = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file temporarily to the disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())  # Write the content of the uploaded file to disk
                temp_file_path = tmp_file.name  # Get the path of the temporary file

            # Load the PDF from the saved temporary file
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())  # Load all the documents from the file

        # Text splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)

        # Generate FAISS vector store with embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# File uploader for real-time PDF upload
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Process Uploaded PDFs") and uploaded_files:
    vector_embedding(uploaded_files)
    st.write("Vector Store DB is ready!")

# Input for the question
prompt1 = st.text_input("Enter your question based on the documents")

# Processing the question
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start

    # Display the response
    st.write(f"Response time: {response_time:.2f} seconds")
    st.write(response['answer'])

    # With a Streamlit expander to show similar documents
    with st.expander("Document Similarity Search"):
        # Display the relevant chunks from the documents
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("--------------------------------")
