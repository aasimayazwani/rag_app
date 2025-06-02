import streamlit as st
import os, json
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader, PyPDFLoader

# ────────────────────────────────────────────
# 1. Paths & Keys
# ────────────────────────────────────────────
DEFAULT_DATA_DIR = "data_files"
UPLOAD_DIR       = "uploaded_files"
FAISS_DIR        = "faiss_index"

for p in (DEFAULT_DATA_DIR, UPLOAD_DIR, FAISS_DIR):
    os.makedirs(p, exist_ok=True)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]  = os.getenv("GROQ_API_KEY")

# ────────────────────────────────────────────
# 2. LLM + Prompt
# ────────────────────────────────────────────
llm      = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="Llama3-8b-8192")
embedder = OpenAIEmbeddings()
prompt   = ChatPromptTemplate.from_template("""
Answer the question strictly from the provided context.
<context>
{context}
</context>
Question: {input}
""")

# ────────────────────────────────────────────
# 3. Session State
# ────────────────────────────────────────────
if "vectors" not in st.session_state: st.session_state.vectors = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ────────────────────────────────────────────
# 4. Build FAISS Index (PDF + CSV)
# ────────────────────────────────────────────
def build_index_from_folder(folder: str):
    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(".csv"):
            docs.extend(CSVLoader(file_path=path, encoding="utf-8").load())
        elif file.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, embedder)

# ────────────────────────────────────────────
# 5. Load/Create Vector Index
# ────────────────────────────────────────────
faiss_file = os.path.join(FAISS_DIR, "index.faiss")
if st.session_state.vectors is None:
    if os.path.exists(faiss_file):
        st.session_state.vectors = FAISS.load_local(FAISS_DIR, embedder)
    else:
        index = build_index_from_folder(DEFAULT_DATA_DIR)
        if index:
            index.save_local(FAISS_DIR)
            st.session_state.vectors = index

# ────────────────────────────────────────────
# 6. UI Header
# ────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot | CSV + PDF", layout="wide")
st.title("🧠 Document Q&A Bot (CSV + PDF)")

# ────────────────────────────────────────────
# 7. Sidebar – Upload & Controls
# ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Indexed Files")
    all_files = sorted(
        [os.path.join(DEFAULT_DATA_DIR, f) for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith((".csv", ".pdf"))] +
        [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith((".csv", ".pdf"))]
    )
    if all_files:
        for f in all_files:
            st.markdown(f"- {os.path.basename(f)}")
    else:
        st.info("No files indexed yet.")

    with st.expander("➕ Upload CSV or PDF"):
        uploads = st.file_uploader("Upload files", type=["csv", "pdf"], accept_multiple_files=True)

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history.clear()
        st.success("History cleared.")

    if st.session_state.chat_history:
        data = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button("⬇️ Download Chat Log", data, file_name="chat_history.json")

# ────────────────────────────────────────────
# 8. Process Uploads
# ────────────────────────────────────────────
if uploads:
    existing = set(os.listdir(DEFAULT_DATA_DIR)) | set(os.listdir(UPLOAD_DIR))
    new_docs, skipped = [], []

    for file in uploads:
        if file.name in existing:
            skipped.append(file.name)
            continue
        save_path = os.path.join(UPLOAD_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
        if file.name.lower().endswith(".csv"):
            new_docs.extend(CSVLoader(save_path, encoding="utf-8").load())
        elif file.name.lower().endswith(".pdf"):
            new_docs.extend(PyPDFLoader(save_path).load())

    if new_docs:
        chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100).split_documents(new_docs)
        new_index = FAISS.from_documents(chunks, embedder)
        if st.session_state.vectors:
            st.session_state.vectors.merge_from(new_index)
        else:
            st.session_state.vectors = new_index
        st.session_state.vectors.save_local(FAISS_DIR)
        st.success("✅ Index updated with new files.")

    if skipped:
        st.warning(f"⚠️ Skipped duplicates: {', '.join(skipped)}")

# ────────────────────────────────────────────
# 9. Chat Interface
# ────────────────────────────────────────────
if st.session_state.vectors:
    query = st.chat_input("Ask a question about your data")
    if query:
        chain = create_retrieval_chain(
            st.session_state.vectors.as_retriever(),
            create_stuff_documents_chain(llm, prompt),
        )
        with st.spinner("Thinking..."):
            res = chain.invoke({"input": query})
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": query,
            "answer": res["answer"],
        })
else:
    st.info("🚩 No documents indexed yet. Upload files from sidebar.")

# ────────────────────────────────────────────
# 10. Display Chat
# ────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {msg['timestamp']}**\n\n{msg['question']}")
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
