import streamlit as st
import os
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
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

# === Setup directories ===
BASE_DIR = "rag_app_data"
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load environment variables ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# === LLM and prompt setup ===
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="Llama3-8b-8192")
embedder = OpenAIEmbeddings()
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
<context>
{context}
</context>
Question: {input}
"""
)

# === Session state init ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# === Attempt to load FAISS if exists ===
if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
    st.session_state.vectors = FAISS.load_local(FAISS_DIR, embedder)

# === Streamlit page config ===
st.set_page_config(page_title="RAG Chatbot with File Upload", layout="wide")
st.title("📄 RAG Chatbot with Groq | PDF + CSV | Chat History")

# === Upload files ===
uploaded_files = st.file_uploader("Upload files (PDF or CSV)", type=["pdf", "csv"], accept_multiple_files=True)

# === Process uploaded files ===
if uploaded_files:
    new_docs, skipped = [], []
    existing_files = set(os.listdir(UPLOAD_DIR))

    for file in uploaded_files:
        if file.name in existing_files:
            skipped.append(file.name)
            continue

        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.name.endswith(".csv"):
            loader = CSVLoader(file_path=path, encoding="utf-8")
        else:
            continue

        new_docs.extend(loader.load())

    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(new_docs)
        new_index = FAISS.from_documents(chunks, embedder)

        if st.session_state.vectors:
            st.session_state.vectors.merge_from(new_index)
        else:
            st.session_state.vectors = new_index

        st.session_state.vectors.save_local(FAISS_DIR)
        st.success("✅ Documents indexed.")

    if skipped:
        st.warning(f"⚠️ Skipped (already uploaded): {', '.join(skipped)}")

# === Clear Chat ===
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history.clear()
    st.success("History cleared.")

# === Download Chat Log ===
if st.session_state.chat_history:
    history_json = json.dumps(st.session_state.chat_history, indent=2)
    st.download_button("⬇️ Download Chat Log", history_json, file_name="chat_history.json")

# === Chat box ===
question = st.chat_input("Ask a question about your documents")

if question:
    if st.session_state.vectors:
        chain = create_retrieval_chain(
            st.session_state.vectors.as_retriever(),
            create_stuff_documents_chain(llm, prompt),
        )
        with st.spinner("Generating response..."):
            res = chain.invoke({"input": question})
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "question": question,
                "answer": res["answer"]
            })
    else:
        st.warning("🚩 No documents uploaded yet. Please upload PDFs or CSVs.")

# === Chat log display ===
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {msg['timestamp']}**\n\n{msg['question']}")
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
