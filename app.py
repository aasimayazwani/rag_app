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
from langchain_community.document_loaders import PyPDFLoader

# ╭─────────────────────────────────────────╮
# │ 1. Folders & environment               │
# ╰─────────────────────────────────────────╯
DEFAULT_DOC_DIR = "research_papers"   # pre-loaded docs
UPLOAD_DIR      = "uploaded_docs"     # user uploads
FAISS_DIR       = "faiss_index"       # persistent index

for d in (DEFAULT_DOC_DIR, UPLOAD_DIR, FAISS_DIR):
    os.makedirs(d, exist_ok=True)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]  = os.getenv("GROQ_API_KEY")

# ╭─────────────────────────────────────────╮
# │ 2. LLM & prompt                        │
# ╰─────────────────────────────────────────╯
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
Answer the question strictly from the provided context.
<context>
{context}
</context>
Question: {input}
"""
)

embedder = OpenAIEmbeddings()

# ╭─────────────────────────────────────────╮
# │ 3. Session-state defaults              │
# ╰─────────────────────────────────────────╯
if "vectors"      not in st.session_state: st.session_state.vectors = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ╭─────────────────────────────────────────╮
# │ 4. Helper: build vector store once     │
# ╰─────────────────────────────────────────╯
def build_index_from_folder(folder: str):
    """Create FAISS index from every PDF in a folder."""
    docs = []
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join(folder, file)).load())

    if not docs:  # empty folder, return None
        return None

    chunks   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(chunks, embedder)

# ╭─────────────────────────────────────────╮
# │ 5. Start-up: load or create FAISS      │
# ╰─────────────────────────────────────────╯
faiss_path = os.path.join(FAISS_DIR, "index.faiss")
if st.session_state.vectors is None:
    if os.path.exists(faiss_path):
        st.session_state.vectors = FAISS.load_local(FAISS_DIR, embedder)
    else:
        # Build from default docs and persist
        base_index = build_index_from_folder(DEFAULT_DOC_DIR)
        if base_index:           # only save if we actually found PDFs
            base_index.save_local(FAISS_DIR)
            st.session_state.vectors = base_index

# ╭─────────────────────────────────────────╮
# │ 6. UI header                           │
# ╰─────────────────────────────────────────╯
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Document Chatbot  |  Groq + Llama3")

# ╭─────────────────────────────────────────╮
# │ 7. Sidebar – doc list & tools          │
# ╰─────────────────────────────────────────╯
with st.sidebar:
    st.markdown("## 📂 Knowledge-base PDFs")

    all_known_pdfs = sorted(
        [os.path.join("research_papers", f) for f in os.listdir(DEFAULT_DOC_DIR) if f.endswith(".pdf")]
        + [os.path.join("uploaded_docs",   f) for f in os.listdir(UPLOAD_DIR)    if f.endswith(".pdf")]
    )
    if all_known_pdfs:
        for f in all_known_pdfs:
            st.markdown(f"- {os.path.basename(f)}")
    else:
        st.info("No PDFs found yet.")

    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history.clear()
        st.success("Chat history cleared.")

    if st.session_state.chat_history:
        data = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button("⬇️ Download Chat Log", data, file_name="chat_history.json")

# ╭─────────────────────────────────────────╮
# │ 8. Upload & embed – duplicates safe    │
# ╰─────────────────────────────────────────╯
uploads = st.file_uploader(
    "📎 Upload additional PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if uploads:
    new_docs, skipped = [], []
    # full set of filenames already present (default + previous uploads)
    existing_names = {f for f in os.listdir(DEFAULT_DOC_DIR) if f.endswith(".pdf")}
    existing_names |= {f for f in os.listdir(UPLOAD_DIR)      if f.endswith(".pdf")}

    for file in uploads:
        if file.name in existing_names:
            skipped.append(file.name)
            continue

        save_to = os.path.join(UPLOAD_DIR, file.name)
        with open(save_to, "wb") as f:
            f.write(file.read())

        new_docs.extend(PyPDFLoader(save_to).load())
        existing_names.add(file.name)

    if new_docs:
        with st.spinner("🔄 Embedding new PDFs…"):
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(new_docs)

            new_store = FAISS.from_documents(chunks, embedder)

            if st.session_state.vectors:
                st.session_state.vectors.merge_from(new_store)
            else:
                st.session_state.vectors = new_store

            st.session_state.vectors.save_local(FAISS_DIR)
            st.success("✅ Vector store updated with new PDFs!")

    if skipped:
        st.warning(f"⚠️ Skipped duplicate(s): {', '.join(skipped)}")

# ╭─────────────────────────────────────────╮
# │ 9. Chat – ask question if vectors ready│
# ╰─────────────────────────────────────────╯
if st.session_state.vectors:
    user_q = st.chat_input("Ask a question about your PDFs")
    if user_q:
        chain = create_retrieval_chain(
            st.session_state.vectors.as_retriever(),
            create_stuff_documents_chain(llm, prompt),
        )
        with st.spinner("Generating answer…"):
            res = chain.invoke({"input": user_q})

        st.session_state.chat_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question":  user_q,
                "answer":    res["answer"],
            }
        )
else:
    st.info("🚩 No PDFs indexed yet. Add some to begin chatting.")

# ╭─────────────────────────────────────────╮
# │10. Display chat messages               │
# ╰─────────────────────────────────────────╯
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**🕒 {item['timestamp']}**\n\n{item['question']}")
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
