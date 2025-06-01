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

# Prompt setup
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
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# --- APP UI ---
st.set_page_config(page_title="RAG Q&A with Groq", layout="wide")
st.title("💬 RAG Document Chatbot (Groq + Llama3)")

# Clear button
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

# Chat history first (scrollable, expandable)
if st.session_state.chat_history:
    st.markdown("### 📜 Previous Conversations")
    for idx, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"🧑 Q{idx+1}: {q}", expanded=False):
            st.markdown(f"🤖 **Answer:**\n\n{a}")

st.markdown("---")

# Embed PDFs
if st.button("📚 Embed Research PDFs"):
    with st.spinner("Embedding documents..."):
        st.session_state.embeddings = OpenAIEmbeddings()
        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs[:50])
        st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)
        st.success("✅ Vector embeddings are ready.")

# Chat input form
with st.form("query_form", clear_on_submit=True):
    user_prompt = st.text_input("🔍 Ask a question based on the documents", key="chat_input")
    submitted = st.form_submit_button("Send")

if submitted and user_prompt and st.session_state.vectors:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Thinking..."):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed = time.process_time() - start

        # Save and clear input
        st.session_state.chat_history.append((user_prompt, response['answer']))
        st.session_state.last_question = user_prompt

    # Scroll to top workaround
    st.experimental_rerun()

# Show current answer after rerun
if st.session_state.last_question:
    st.markdown("### 🤖 Current Answer")
    st.markdown(st.session_state.chat_history[-1][1])
    st.markdown("---")

    with st.expander("📄 Matched Document Chunks"):
        for doc in response['context']:
            st.write(doc.page_content)
            st.markdown("---")
