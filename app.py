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

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="Llama3-8b-8192")

# Prompt template
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

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App title
st.title("📚 RAG Document Q&A with Groq & Llama3")

# Optional: Clear chat
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

# === 🧠 Show Chat History (Chronological - oldest first) ===
if st.session_state.chat_history:
    st.markdown("## 🕘 Chat History")
    for idx, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"🧑 Q{idx+1}: {q}", expanded=False):
            st.markdown(f"**🤖 A{idx+1}:** {a}")

# === 📥 Embed PDFs if not already done ===
if st.button("📚 Embed Research PDFs"):
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Your PDF folder
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.success("✅ Document embeddings created.")

# === 🔍 Ask a question ===
user_prompt = st.text_input("Ask a question based on the documents")

if user_prompt and st.session_state.vectors:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    elapsed = time.process_time() - start

    # Store chat
    st.session_state.chat_history.append((user_prompt, response['answer']))

    # Show current response
    st.subheader("🤖 Current Answer")
    st.write(response['answer'])

    # Show source documents
    with st.expander("📄 Matching Document Chunks"):
        for doc in response['context']:
            st.write(doc.page_content)
            st.markdown("---")