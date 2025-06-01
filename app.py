import streamlit as st
import os
import shutil
import time
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# === Directories ===
BASE_DIR = "/mnt/data/rag_app_data"
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load env keys ===
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# === LLM + Prompt ===
llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
<context>
{context}
</context>
Question: {input}
""")

# === Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# === Load FAISS Index If Available ===
if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
    st.session_state.vectors = FAISS.load_local(FAISS_DIR, OpenAIEmbeddings())

# === Page Config ===
st.set_page_config(page_title="RAG Chatbot with Upload", layout="wide")
st.title("📄 RAG Chatbot with Groq + File Upload + Timestamps")

# === Upload PDFs ===
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing and embedding PDFs..."):
        all_docs = []
        for file in uploaded_files:
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings()
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectors.save_local(FAISS_DIR)
        st.success("✅ Documents embedded and saved!")

# === Clear Chat Button ===
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

# === Download Chat History ===
if st.session_state.chat_history:
    history_json = json.dumps(st.session_state.chat_history, indent=2)
    st.download_button("⬇️ Download Chat Log", history_json, file_name="chat_history.json")

# === Chat Interface ===
if st.session_state.vectors:
    user_input = st.chat_input("Ask a question about your uploaded documents")

    if user_input:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Generating response..."):
            response = retrieval_chain.invoke({"input": user_input})
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save message with timestamp
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "question": user_input,
                "answer": response["answer"]
            })

# === Show Chat Messages ===
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**🕒 {msg['timestamp']}**")
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
