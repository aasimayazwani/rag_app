import streamlit as st
import os, json, csv
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

# ╭─────────────────────────────────────────╮
# │ 1. Folders & Keys                      │
# ╰─────────────────────────────────────────╯
DEFAULT_DATA_DIR = "data_csv"       # starter CSVs go here
UPLOAD_DIR       = "uploaded_csv"    # run-time uploads
FAISS_DIR        = "faiss_index"
for p in (DEFAULT_DATA_DIR, UPLOAD_DIR, FAISS_DIR):
    os.makedirs(p, exist_ok=True)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]  = os.getenv("GROQ_API_KEY")

# ╭─────────────────────────────────────────╮
# │ 2. LLM + Prompt                        │
# ╰─────────────────────────────────────────╯
llm      = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"],
                    model_name="Llama3-8b-8192")
embedder = OpenAIEmbeddings()
prompt   = ChatPromptTemplate.from_template(
    """
Answer the question strictly from the provided context.
<context>
{context}
</context>
Question: {input}
"""
)

# ╭─────────────────────────────────────────╮
# │ 3. Session State                       │
# ╰─────────────────────────────────────────╯
if "vectors"      not in st.session_state: st.session_state.vectors = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ╭─────────────────────────────────────────╮
# │ 4. Helper – build FAISS from CSV rows  │
# ╰─────────────────────────────────────────╯
def build_index_from_csv_folder(folder: str):
    docs = []
    for file in os.listdir(folder):
        if file.lower().endswith(".csv"):
            loader = CSVLoader(file_path=os.path.join(folder, file), encoding="utf-8")
            docs.extend(loader.load())           # each row becomes one Document
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, embedder)

# ╭─────────────────────────────────────────╮
# │ 5. Load or Create FAISS index          │
# ╰─────────────────────────────────────────╯
faiss_file = os.path.join(FAISS_DIR, "index.faiss")
if st.session_state.vectors is None:
    if os.path.exists(faiss_file):
        st.session_state.vectors = FAISS.load_local(FAISS_DIR, embedder)
    else:
        base = build_index_from_csv_folder(DEFAULT_DATA_DIR)
        if base:
            base.save_local(FAISS_DIR)
            st.session_state.vectors = base

# ╭─────────────────────────────────────────╮
# │ 6. UI Header                           │
# ╰─────────────────────────────────────────╯
st.set_page_config(page_title="CSV RAG Chatbot", layout="wide")
st.title("📊 CSV Data Chatbot | Groq + Llama3")

# ╭─────────────────────────────────────────╮
# │ 7. Sidebar – file list & controls      │
# ╰─────────────────────────────────────────╯
with st.sidebar:
    st.markdown("### 🗂 Indexed CSV Files")
    csv_files = sorted(
        [os.path.join("data_csv",    f) for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith(".csv")]
      + [os.path.join("uploaded_csv", f) for f in os.listdir(UPLOAD_DIR)      if f.endswith(".csv")]
    )
    if csv_files:
        for f in csv_files:
            st.markdown(f"- {os.path.basename(f)}")
    else:
        st.info("No CSVs indexed yet.")

    # Upload expander
    with st.expander("➕ Upload / add more CSVs"):
        uploads = st.file_uploader(
            "Select CSV files",
            type="csv",
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="csv_uploader",
        )

    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history.clear()
        st.success("History cleared.")
    if st.session_state.chat_history:
        data = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button("⬇️ Download Chat Log", data, file_name="chat_history.json")

# ╭─────────────────────────────────────────╮
# │ 8. Handle Uploads (duplicates safe)    │
# ╰─────────────────────────────────────────╯
if uploads:
    existing = {f for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith(".csv")}
    existing |= {f for f in os.listdir(UPLOAD_DIR)      if f.endswith(".csv")}

    new_docs, skipped = [], []
    for file in uploads:
        if file.name in existing:
            skipped.append(file.name)
            continue
        save_p = os.path.join(UPLOAD_DIR, file.name)
        with open(save_p, "wb") as f:
            f.write(file.read())
        new_docs.extend(CSVLoader(save_p, encoding="utf-8").load())
        existing.add(file.name)

    if new_docs:
        chunks    = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=100).split_documents(new_docs)
        new_store = FAISS.from_documents(chunks, embedder)

        if st.session_state.vectors:
            st.session_state.vectors.merge_from(new_store)
        else:
            st.session_state.vectors = new_store

        st.session_state.vectors.save_local(FAISS_DIR)
        st.success("✅ Vector store updated with new CSV data.")
    if skipped:
        st.warning(f"⚠️ Duplicate(s) skipped: {', '.join(skipped)}")

# ╭─────────────────────────────────────────╮
# │ 9. Chat Interface                      │
# ╰─────────────────────────────────────────╯
if st.session_state.vectors:
    question = st.chat_input("Ask a question about your CSV data")
    if question:
        chain = create_retrieval_chain(
            st.session_state.vectors.as_retriever(),
            create_stuff_documents_chain(llm, prompt),
        )
        with st.spinner("Generating answer…"):
            res = chain.invoke({"input": question})
        st.session_state.chat_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question":  question,
                "answer":    res["answer"],
            }
        )
else:
    st.info("🚩 No CSV data indexed yet. Add files from the sidebar.")

# ╭─────────────────────────────────────────╮
# │10. Display conversation                │
# ╰─────────────────────────────────────────╯
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {msg['timestamp']}**\n\n{msg['question']}")
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
