import streamlit as st
import tempfile
import os
import base64
import asyncio
import time
import sys
from agent_RC2 import run_conversation  # Import the voice assistant function

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore


# 📌 Upload PDF and Store in AstraDB
def vector_datastore_upload(pdf_path):
    log_dir = "logs_astraDB"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"astraDB_log_{timestamp}.txt")

    with open(log_filename, 'w') as log_file:
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        vstore_pdf = AstraDBVectorStore(
            collection_name="uploaded",
            embedding=embedding,
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        print("Connected to AstraDB for user uploads")  # Debugging print

        # 🚨 **CLEAR OLD DOCUMENTS BEFORE UPLOADING NEW ONES** 🚨
        vstore_pdf.clear()
        print("Cleared old documents from AstraDB")

        loader_pdf = PyPDFLoader(pdf_path)
        docs_pdf = loader_pdf.load()
        documents_pdf = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(docs_pdf)

        inserted_ids_pdf = vstore_pdf.add_documents(documents_pdf)
        print("Inserted new documents into AstraDB")
        log_file.write(f"\nInserted {len(inserted_ids_pdf)} documents.")

    return "✅ Document uploaded and processed!"


# 📌 Streamlit App Configuration
st.set_page_config(page_title="Luna - Legal RAG Assistant", layout="wide")

st.title("🗣️ Luna - Legal Compliance & Risk Assistant")

# 📌 Session State Variables
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None

if "log_messages" not in st.session_state:
    st.session_state.log_messages = []  # Store live chat logs
    st.session_state.log_filename = None

# 📌 PDF Upload
uploaded_file = st.file_uploader("📂 Upload a legal document (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.session_state.uploaded_file_path = temp_file_path
    st.success(vector_datastore_upload(st.session_state.uploaded_file_path))


# 📌 Display Uploaded PDF
def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'


if st.session_state.uploaded_file_path:
    st.subheader("📄 Uploaded Document Preview")
    st.markdown(display_pdf(st.session_state.uploaded_file_path), unsafe_allow_html=True)


# 📌 Sidebar for Live Logs
st.sidebar.title("📌 Live Chat History")
st.sidebar.caption("🔹 Memories")
sidebar_container = st.sidebar.empty()


# 📌 StreamLogger: Capture and Display Logs
class StreamLogger:
    def __init__(self, log_container, log_file_path):
        self.terminal = sys.stdout  # Store original stdout
        self.log_container = log_container
        self.buffer = ""  # Temporary buffer
        self.log_file = open(log_file_path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.buffer += message
        self.log_file.write(message)

        if "\n" in message:
            message = self.buffer.strip()
            if message and message not in st.session_state.log_messages:
                st.session_state.log_messages.append(message)  # Prevent duplicates
                self.buffer = ""  # Reset buffer
                self.update_sidebar()  # Update sidebar display

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        self.log_file.close()  # Close file properly

    def update_sidebar(self):
        self.log_container.empty()
        with self.log_container.container():
            for entry in st.session_state.log_messages:
                st.markdown(f"📝 {entry}")


# 📌 Voice Assistant Activation
if st.button("🎙️ Activate Voice Assistant"):
    st.info("🎤 Voice Assistant Activated. Speak your question...")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"chatbot_log_{timestamp}.txt")
    st.session_state.log_filename = log_filename

    sys.stdout = StreamLogger(sidebar_container, log_filename)

    try:
        asyncio.run(run_conversation())
        print("\nConversation completed.")  # Log completion
    except Exception as e:
        print(f"\nError: {e}")

    sys.stdout = sys.__stdout__

    st.success("✅ Conversation completed. Check the sidebar for chat history.")
