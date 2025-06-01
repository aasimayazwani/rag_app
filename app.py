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

# Load env vars
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# LLM and Prompt
llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="Llama3-8b-8192")

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

# Session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Title
st.set_page_config(page_title="Groq RAG Chatbot", layout="wide")
st.title("💬 RAG Document Chatbot (Groq + Llama3)")

# Sidebar: Clear
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.success("Chat history cleared.")

# Embed PDFs
if st.button("📚 Embed Research PDFs"):
    with st.spinner("Embedding documents..."):
        embeddings = OpenAIEmbeddings()
        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs[:50])
        st.session_state.vectors = FAISS.from_documents(splits, embeddings)
        st.success("✅ Vector embeddings ready.")

# Chat History
if st.session_state.chat_history:
    st.markdown("### 🕘 Chat History")
    for idx, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"🧑 Q{idx+1}: {q}", expanded=False):
            st.markdown(f"**🤖 A{idx+1}:** {a}")
    st.markdown("---")

# Chat Form (auto-clears)
with st.form("query_form", clear_on_submit=True):
    user_prompt = st.text_input("🔍 Ask a question from the documents")
    submitted = st.form_submit_button("Send")

# Run chat only if input is submitted
if submitted and user_prompt and st.session_state.vectors:
    with st.spinner("Generating answer..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed = time.process_time() - start

        st.session_state.chat_history.append((user_prompt, response['answer']))
        st.session_state.last_response = response  # Store last for current view

# Show most recent response (if any)
if st.session_state.last_response:
    st.markdown("### 🤖 Current Answer")
    st.write(st.session_state.last_response['answer'])

    with st.expander("📄 Matched Document Chunks"):
        for doc in st.session_state.last_response['context']:
            st.write(doc.page_content)
            st.markdown("---")
