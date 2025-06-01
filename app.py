import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize LLM
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

# Prompt Template WITH history placeholder
prompt = ChatPromptTemplate.from_template(
    """
    You are an AI research assistant. Answer the question based on the context and conversation history.
    
    <history>
    {history}
    </history>

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Function to embed documents
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("🧠 RAG Chatbot with Memory - LLaMA3 via Groq")

if st.button("Create Document Embedding"):
    create_vector_embedding()
    st.success("✅ Vector database is ready!")

user_prompt = st.text_input("Ask something from the documents:")

if user_prompt:
    # Append user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Convert chat history into a string
    history_text = ""
    for msg in st.session_state.chat_history:
        prefix = "User" if msg["role"] == "user" else "AI"
        history_text += f"{prefix}: {msg['content']}\n"

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt, 'history': history_text})
    elapsed = time.process_time() - start

    # Save assistant reply to history
    st.session_state.chat_history.append({"role": "ai", "content": response['answer']})

    # Show the answer
    st.markdown(f"**🧠 LLaMA3:** {response['answer']}")
    st.caption(f"⏱️ Response time: {elapsed:.2f}s")

    with st.expander("🔍 Retrieved Document Chunks"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.markdown("---")

# Optional: Clear history
if st.button("🔁 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
