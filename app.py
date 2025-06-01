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
from langchain_community.document_loaders import PyPDFLoader

# ╭───────────────────────────────────╮
# │ 1. Folders & ENV keys            │
# ╰───────────────────────────────────╯
FAISS_DIR   = "faiss_index"
UPLOAD_DIR  = "uploaded_docs"
os.makedirs(FAISS_DIR,  exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]  = os.getenv("GROQ_API_KEY")

# ╭───────────────────────────────────╮
# │ 2. LLM + Prompt                  │
# ╰───────────────────────────────────╯
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the question strictly from the provided context.
<context>
{context}
</context>
Question: {input}
"""
)

# ╭───────────────────────────────────╮
# │ 3. Session-state defaults        │
# ╰───────────────────────────────────╯
if "vectors"      not in st.session_state: st.session_state.vectors = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# Load existing FAISS index if present
if (
    st.session_state.vectors is None
    and os.path.exists(os.path.join(FAISS_DIR, "index.faiss"))
):
    st.session_state.vectors = FAISS.load_local(FAISS_DIR, OpenAIEmbeddings())

# ╭───────────────────────────────────╮
# │ 4. Page layout                   │
# ╰───────────────────────────────────╯
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Document Chatbot  |  Groq + Llama3")

# ── Sidebar – utilities & file list ─────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Options")

    # List already-uploaded PDFs
    existing_pdfs = sorted(f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf"))
    if existing_pdfs:
        st.markdown("#### 📂 Uploaded PDFs")
        for f in existing_pdfs:
            st.markdown(f"- {f}")

    # Chat-history tools
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history.clear()
        st.success("Chat log cleared.")

    if st.session_state.chat_history:
        log_json = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button("⬇️ Download Chat Log", log_json, file_name="chat_history.json")

# ╭───────────────────────────────────╮
# │ 5. PDF upload & embedding        │
# ╰───────────────────────────────────╯
uploaded_files = st.file_uploader(
    "📎 Upload PDF files (you can add more at any time)",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    new_docs, skipped = [], []
    # gather filenames already on disk (for duplicate detection)
    existing_pdfs_set = set(existing_pdfs)

    for file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, file.name)
        if file.name in existing_pdfs_set:               # duplicate found
            skipped.append(file.name)
            continue

        # Save the new file
        with open(save_path, "wb") as f:
            f.write(file.read())

        # Load PDF into LangChain docs
        new_docs.extend(PyPDFLoader(save_path).load())
        existing_pdfs_set.add(file.name)                 # add to set for future checks

    if new_docs:
        with st.spinner("🔄 Embedding new documents…"):
            chunks      = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(new_docs)
            embeddings  = OpenAIEmbeddings()
            new_store   = FAISS.from_documents(chunks, embeddings)

            if st.session_state.vectors:                 # merge with existing index
                st.session_state.vectors.merge_from(new_store)
            else:
                st.session_state.vectors = new_store

            st.session_state.vectors.save_local(FAISS_DIR)
            st.success("✅ Vector store updated!")

    if skipped:
        st.warning(f"⚠️ Skipped duplicate file(s): {', '.join(skipped)}")

# ╭───────────────────────────────────╮
# │ 6. Chat input / retrieval-QA     │
# ╰───────────────────────────────────╯
if st.session_state.vectors:
    user_msg = st.chat_input("Ask a question about your uploaded PDFs")

    if user_msg:
        chain = create_retrieval_chain(
            st.session_state.vectors.as_retriever(),
            create_stuff_documents_chain(llm, prompt),
        )
        with st.spinner("Generating answer…"):
            result = chain.invoke({"input": user_msg})

        st.session_state.chat_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question":  user_msg,
                "answer":    result["answer"],
            }
        )

# ╭───────────────────────────────────╮
# │ 7. Display chat messages         │
# ╰───────────────────────────────────╯
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {entry['timestamp']}**\n\n{entry['question']}")
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
