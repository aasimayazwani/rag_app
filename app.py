import streamlit as st
import os
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

# === Setup ===
FAISS_DIR = "faiss_index"
UPLOAD_DIR = "uploaded_docs"
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# === LLM Setup ===
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

# === Load FAISS if available ===
if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")) and st.session_state.vectors is None:
    st.session_state.vectors = FAISS.load_local(FAISS_DIR, OpenAIEmbeddings())

# === App Layout ===
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Document Chatbot (Groq + Llama3)")

# === Upload PDFs ===
uploaded_files = st.file_uploader("📎 Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Embedding uploaded PDFs..."):
        docs = []
        for file in uploaded_files:
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectors.save_local(FAISS_DIR)

        st.success("✅ Documents embedded and vector store updated.")

# === Tools ===
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

with col2:
    if st.session_state.chat_history:
        chat_json = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button("⬇️ Download Chat Log", chat_json, file_name="chat_history.json")

# === Ask Question ===
if st.session_state.vectors:
    user_input = st.chat_input("Ask a question about your uploaded documents")

    if user_input:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Generating answer..."):
            response = retrieval_chain.invoke({"input": user_input})
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Store message
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "question": user_input,
                "answer": response["answer"]
            })

# === Show Chat Messages ===
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {msg['timestamp']}**")
        st.markdown(msg["question"])
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
