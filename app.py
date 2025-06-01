import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load .env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Init LLM
llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Session State Setup
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Page Config
st.set_page_config(page_title="Groq RAG Chatbot", layout="wide")

# === ✨ Custom Styling ===
st.markdown("""
    <style>
        .big-font { font-size:22px; font-weight:600; }
        .chat-bubble-user {
            background-color: #2c2f33;
            padding: 1rem;
            border-radius: 1rem;
            margin-bottom: 0.5rem;
            color: #f1f1f1;
        }
        .chat-bubble-bot {
            background-color: #404552;
            padding: 1rem;
            border-radius: 1rem;
            margin-bottom: 1rem;
            color: #f1f1f1;
        }
        button[kind="primary"] {
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# === 🧠 Sidebar: Vector & History Reset ===
with st.sidebar:
    st.markdown("## ⚙️ Options")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.success("Chat history cleared.")
    if st.button("📚 Embed PDFs"):
        with st.spinner("Embedding documents..."):
            embeddings = OpenAIEmbeddings()
            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs[:50])
            st.session_state.vectors = FAISS.from_documents(splits, embeddings)
            st.success("✅ Documents embedded.")

# === 💬 Title ===
st.title("🤖 RAG Document Q&A Chatbot")

# === 🕘 Chat History Display (Chronological, on top) ===
if st.session_state.chat_history:
    st.markdown("### 📜 Chat History")
    for idx, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"🧑‍💻 Q{idx+1}: {q}", expanded=False):
            st.markdown(f'<div class="chat-bubble-user"><strong>🧑 You:</strong><br>{q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-bot"><strong>🤖 Llama3:</strong><br>{a}</div>', unsafe_allow_html=True)

st.markdown("---")

# === 🔍 Chat Input Box ===
with st.form("query_form", clear_on_submit=True):
    user_prompt = st.text_input("Ask a question from your documents")
    submitted = st.form_submit_button("Send")

# === 🚀 If submitted, answer and store ===
if submitted and user_prompt and st.session_state.vectors:
    with st.spinner("Thinking..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'input': user_prompt})

        # Save new entry
        st.session_state.chat_history.append((user_prompt, response['answer']))
        st.session_state.last_response = response

# === 🧠 Display Last Answer ===
if st.session_state.last_response:
    st.markdown("### 🤖 Current Answer")
    st.markdown(f'<div class="chat-bubble-bot">{st.session_state.last_response["answer"]}</div>', unsafe_allow_html=True)

    with st.expander("📄 Matched Document Chunks"):
        for doc in st.session_state.last_response['context']:
            st.write(doc.page_content)
            st.markdown("---")
