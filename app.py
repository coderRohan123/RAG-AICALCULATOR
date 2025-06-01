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


# Sidebar Navigation
menu = ["🏠 Home", "✍️ Delineate", "🔍 Decipher", "ℹ️ About"]
choice = st.sidebar.selectbox("Navigation", menu)

# Sidebar Content Filling
if choice == "🏠 Home":
    st.sidebar.write("Steps to proceed with your research:")
    st.sidebar.write("1. Upload your research papers.")
    st.sidebar.write("2. Process them to create embeddings.")
    st.sidebar.write("3. Ask any questions or explore the math equation solver.")
    st.sidebar.write("4. Navigate to 'Delineate' for PDF processing or 'Decipher' for visual equation solving.")

elif choice == "✍️ Delineate":
    st.sidebar.title("How to use Delineate:")
    st.sidebar.write("1. Upload your PDF files.")
    st.sidebar.write("2. Click the 'Process Uploaded PDFs' button.")
    st.sidebar.write("3. Once processed, ask questions related to the content of your papers.")
    st.sidebar.write("4. View document similarity and explore the embedded chunks.")

elif choice == "🔍 Decipher":
    st.sidebar.title("How to use Decipher:")
    st.sidebar.write("1. Draw or input your math equations in the interface.")
    st.sidebar.write("2. Submit your problem to get solutions.")
    st.sidebar.write("3. Explore the process or steps to reach the answer visually.")

elif choice == "ℹ️ About":
    st.sidebar.write("About this project:")
    st.sidebar.write("Developed for PhD-level research assistance.")
    st.sidebar.write("Powered by Llama 3.1 and FAISS for precise academic insights.")

# Home Page
if choice == "🏠 Home":
    st.markdown(
        """
        <h1 style="text-align: center;">
        DELINEATE AND DECIPHER <br>
        "A Rag powered AI platform for Research Paper Analysis and Visual Math Problem Solving"
        </h1>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        @keyframes flicker {
            0%, 18%, 22%, 25%, 53%, 57%, 100% {
                opacity: 1;
            }
            20%, 24%, 55% {
                opacity: 0.4;
            }
            21%, 23%, 56% {
                opacity: 0.7;
            }
        }
        .marquee {
            animation: flicker 1.5s infinite;
            font-size: 24px;
            font-weight: bold;
            color: #FFA500; /* Orange color */
        }
        </style>

        <marquee behavior="scroll" direction="left" scrollamount="6">
            <span class="marquee">Delineate and Decipher: Revolutionizing Research and Problem-Solving</span>
        </marquee>
        """,
        unsafe_allow_html=True
    )
    # Display image and content
    st.markdown(
        """
        ### **Why Use Me?**
        - **Tired of spending hours digging through endless research papers?**
        - **Struggling to quickly locate relevant information from mountains of academic documents?**
        - **Frustrated with complicated math problems that take hours to solve manually?**
        """, unsafe_allow_html=True
    )

    # Add image from the link
    st.image("https://i.imghippo.com/files/kLO6363N.png",
             caption="Delineate and Decipher: Revolutionizing Research and Problem-Solving", use_container_width=False)

    st.markdown(
        """
        **DELINEATE AND DECIPHER** is here to change the way you approach academic research and complex problem-solving. 
        Whether you're a PhD candidate, researcher, or student, this platform is specifically designed to make your life easier and your work more efficient.

        ### **Key Features:**
        - **Delineate**: Seamlessly upload your research papers and let the platform process them into **searchable vector embeddings**. No more endless scrolling—find exactly what you need, when you need it.
        - **Decipher**: Draw or input your toughest math equations, and get **step-by-step visual solutions** powered by cutting-edge AI. It’s like having a math tutor in your pocket.

        ### **Why the Need for This Innovation?** 

        In the modern world of academia and research, time is your most valuable resource. Here's why **Delineate and Decipher** is a game-changer:

        1. **Information Overload:**
           - With the explosion of academic content, it's becoming increasingly challenging to extract useful information from lengthy research papers. Traditional search engines aren't built for complex academic queries. **Delineate** uses advanced natural language processing to **understand your research needs**, extracting answers with precision and speed.

        2. **Complex Problem-Solving:**
           - Math and technical subjects often require deep conceptual understanding and can take hours, if not days, to solve. With **Decipher**, you can draw equations and get **instant feedback** with detailed explanations. The days of struggling with step-by-step solutions are over.

        3. **The Need for Speed in Research:**
           - Traditional methods of document retrieval are **inefficient**. By transforming your papers into **vector embeddings**, the platform allows for fast, targeted searches based on the content itself, not just keywords. It's designed for **efficient, context-driven research**—a massive upgrade over conventional tools.

        ### **The Current Dilemma:**

        Researchers today face a unique dilemma—**too much information, but not enough time** to extract the knowledge they need. Whether you're working on a literature review or solving a challenging mathematical model, the barriers to success often lie in navigating large volumes of data and understanding complex problems quickly.

        ### **Why Delineate and Decipher?**

        **Because it’s the solution to the modern researcher’s biggest challenges:**
        - **Efficiency:** Get instant answers from your research papers, saving time for deeper thinking and creativity.
        - **Precision:** Ask questions about your documents, and the platform delivers **contextually relevant responses**.
        - **Visualization:** Whether it's a math problem or a research inquiry, understanding is more powerful when it's visual. This platform helps you **see the solutions**, not just read them.

        **Ready to explore? Upload your research papers or dive into problem-solving with Decipher!**
        """
    )




# Delineate section
elif choice == "✍️ Delineate":
    st.title("Delineate: Upload and Process Documents")

    # Check if files are already uploaded in session state
    if "uploaded_files" in st.session_state:
        st.write("Previously uploaded files found. You can proceed with questions.")
    else:
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Process Uploaded PDFs") and uploaded_files:
            st.session_state.uploaded_files = uploaded_files  # Store files in session state
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

        # Show document similarity search results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Decipher Page
elif choice == "🔍 Decipher":
    st.title("Decipher: Solve Math Equations Visually")

    st.markdown("""
        **Decipher** is the section where you can draw mathematical equations, and the platform provides step-by-step solutions using advanced AI visualization. 
        Simply input your equation and get a detailed solution.
    """)

    # Embed the localhost page for the math equation solving tool
    st.components.v1.iframe("https://newfrontend-gray.vercel.app", width=800, height=600)

# About Page
elif choice == "ℹ️ About":
    st.title("About:")
    st.markdown("""
        **DELINEATE AND DECIPHER** is an AI-powered platform designed to assist researchers, PhD candidates, and students in analyzing academic papers and solving complex mathematical problems. 

        **What makes it unique:**
        - Uses advanced language models like **Llama 3.1** and **FAISS** for precise academic document retrieval.
        - Efficiently processes research papers, turning them into searchable embeddings.
        - Helps solve math equations with detailed steps, making it perfect for technical problem-solving.

        **Future Enhancements:**
        - Incorporating more advanced mathematical capabilities.
        - Improving support for various academic formats.
        - Expanding the visual tools for document analysis.
    """)


