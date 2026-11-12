"""
DocuMind AI — Multi-PDF RAG Assistant (FAISS edition)

Install:
    pip install -U streamlit langchain langchain-community langchain-openai langchain-text-splitters pypdf faiss-cpu

Windows PowerShell:
    $env:OPENAI_API_KEY="your-api-key"

Run:
    streamlit run app.py
"""

import base64
import os
import tempfile
import uuid

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()
# ----------------------------- Page configuration -----------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="icon.png",
    layout="wide",
)

# ---------------------------------- Styling -----------------------------------



st.markdown("""
<style>

/* Main application background */
.stApp {
    background: #F1F5F9;
    color: #334155;
}

/* Main content area */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Main headings */
h1, h2, h3 {
    color: #1E293B !important;
    font-weight: 700;
}

/* Paragraph and normal text */
p, label, .stMarkdown {
    color: #475569;
}

/* Header title */
.main-title {
    color: #1E293B;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    text-align: center;
}

/* Tagline */
.tagline {
    color: #64748B;
    font-size: 1rem;
    text-align: center;
    margin-bottom: 2rem;
}

/* White content cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px;
}

/* Buttons */
.stButton > button {
    background: #38A9E8;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #0284C7;
    color: #FFFFFF;
    border: none;
}

/* Text input */
.stTextInput input {
    background: #FFFFFF;
    color: #1E293B;
    border: 1px solid #CBD5E1;
    border-radius: 9px;
}

.stTextInput input:focus {
    border-color: #38A9E8;
    box-shadow: 0 0 0 1px #38A9E8;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #F8FAFC;
    border: 1px dashed #7DD3FC;
    border-radius: 12px;
    padding: 15px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #64748B;
}

[data-testid="stMetricValue"] {
    color: #0284C7;
    font-weight: 700;
}

/* Answer section */
.answer-card {
    background: #FFFFFF;
    border-left: 5px solid #38A9E8;
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
    color: #334155;
}

/* Source section */
.source-card {
    background: #E0F2FE;
    border: 1px solid #BAE6FD;
    border-radius: 9px;
    padding: 12px;
    margin: 8px 0;
    color: #334155;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #E2E8F0;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1E293B !important;
}

/* Divider */
hr {
    border-color: #CBD5E1;
}

/* Success messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)




# ------------------------------- Session state --------------------------------

defaults = {
    "vectorstore": None,
    "indexed_chunks": 0,
    "indexed_files": [],
    "total_pages": 0,
    "chat_history": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------- Functions ----------------------------------

def get_api_key():
    """Read the OpenAI API key from the environment variable."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def load_pdfs(uploaded_files):
    """Load every PDF and preserve its original filename and page metadata."""
    all_pages = []
    file_names = []

    for uploaded_file in uploaded_files:
        file_names.append(uploaded_file.name)

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(uploaded_file.getvalue())
                temp_path = temp.name

            pages = PyPDFLoader(temp_path).load()

            for page_doc in pages:
                page_doc.metadata["source"] = uploaded_file.name
                # PyPDFLoader uses zero-based page indexes.
                page_doc.metadata["page"] = int(page_doc.metadata.get("page", 0))

            all_pages.extend(pages)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return all_pages, file_names


def create_faiss_database(pages):
    """Chunk all pages and place every PDF chunk in one FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "No readable text was extracted. This may be a scanned PDF; "
            "OCR support would be needed for image-only PDFs."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # One FAISS vector index containing chunks from all uploaded PDFs.
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


def get_answer(question, vectorstore):
    """Retrieve relevant chunks and generate an answer with source references."""
    retrieved_docs = vectorstore.similarity_search(question, k=5)

    context_sections = []
    for index, doc in enumerate(retrieved_docs, start=1):
        filename = doc.metadata.get("source", "Unknown file")
        page_number = int(doc.metadata.get("page", 0)) + 1
        context_sections.append(
            f"[Source {index}: File: {filename}, Page: {page_number}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_sections)

    prompt = f"""
You are DocuMind AI. Answer the user's question using only the document
excerpts below.

Instructions:
- Use only information found in the excerpts.
- If the answer is not available, say that you could not find it in the
  uploaded documents.
- Do not invent facts, page numbers, or sources.
- Give a clear and concise answer.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    response = llm.invoke(prompt)

    # Keep unique file/page references in retrieval order.
    sources = []
    seen = set()

    for doc in retrieved_docs:
        filename = doc.metadata.get("source", "Unknown file")
        page_number = int(doc.metadata.get("page", 0)) + 1
        key = (filename, page_number)

        if key not in seen:
            seen.add(key)
            sources.append({
                "file": filename,
                "page": page_number,
                "start_index": doc.metadata.get("start_index", "Not available"),
            })

    return response.content, sources


# ----------------------------------- Header -----------------------------------

# Helper function to convert your local image to base64
def get_base64_image(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode()


icon_base64 = get_base64_image("icon.png")

st.markdown(
    f"""
    <div class="hero">
        <h1 style="display: flex; align-items: center; gap: 15px;">
            <img src="data:image/png;base64,{icon_base64}" width="55" style="border-radius: 12px;">
            DocuMind AI
        </h1>
        <p>Ask. Explore. Discover.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------- PDF upload ----------------------------------

st.markdown(
    '<div class="section-heading">1. Upload PDF documents</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Select multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} PDF file(s) selected")

    with st.expander("View selected files", expanded=True):
        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name} · {uploaded_file.size / 1024:.1f} KB")

process_col, reset_col = st.columns([3, 1])

with process_col:
    process_clicked = st.button(
        "⚡ Process & Index",
        type="primary",
        use_container_width=True,
        disabled=not bool(uploaded_files),
    )

with reset_col:
    reset_clicked = st.button("Reset", use_container_width=True)

if reset_clicked:
    st.session_state.vectorstore = None
    st.session_state.indexed_chunks = 0
    st.session_state.indexed_files = []
    st.session_state.total_pages = 0
    st.session_state.chat_history = []
    st.rerun()

if process_clicked:
    if not get_api_key():
        st.error(
            "OPENAI_API_KEY was not found. Set the environment variable "
            "and restart Streamlit."
        )
    else:
        try:
            with st.spinner("Reading PDFs, creating embeddings, and building the FAISS index..."):
                pages, filenames = load_pdfs(uploaded_files)
                vectorstore, chunk_count = create_faiss_database(pages)

                st.session_state.vectorstore = vectorstore
                st.session_state.indexed_chunks = chunk_count
                st.session_state.indexed_files = filenames
                st.session_state.total_pages = len(pages)
                st.session_state.chat_history = []

            st.success("Documents processed and indexed successfully!")
        except Exception as error:
            st.error(f"Processing failed: {error}")


# ------------------------------- Index summary --------------------------------

st.markdown("---")
st.markdown(
    '<div class="section-heading">2. Knowledge base summary</div>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("PDF files indexed", len(st.session_state.indexed_files))
with metric2:
    st.metric("Pages indexed", st.session_state.total_pages)
with metric3:
    st.metric("Chunks indexed", st.session_state.indexed_chunks)

if st.session_state.indexed_files:
    with st.expander("View indexed PDF files"):
        for filename in st.session_state.indexed_files:
            st.write(f"✓ {filename}")


# ------------------------------ Question & answer -----------------------------

st.markdown("---")
st.markdown(
    '<div class="section-heading">3. Ask your question</div>',
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.info("Upload your PDFs and click **Process & Index** to start asking questions.")
else:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="Ask anything about your uploaded PDFs...",
        )
        ask_clicked = st.form_submit_button(
            "Ask DocuMind",
            type="primary",
            use_container_width=True,
        )

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching documents and preparing your answer..."):
                    answer, sources = get_answer(
                        question.strip(),
                        st.session_state.vectorstore,
                    )

                st.session_state.chat_history.append({
                    "question": question.strip(),
                    "answer": answer,
                    "sources": sources,
                })
            except Exception as error:
                st.error(f"Could not generate the answer: {error}")

    # Add @st.cache_data so the image isn't re-read from disk on every message
@st.cache_data
def get_base64_image(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode()


icon_base64 = get_base64_image("icon.png")

# --- Inside your chat history loop ---
for item in reversed(st.session_state.chat_history):

  col_bot, col_answer = st.columns([0.8, 5])

  with col_bot:
    st.markdown(
        f"""
                <div style="text-align:center; padding-top:15px;">
                    <img src="data:image/png;base64,{icon_base64}" 
                         width="85" 
                         height="85" 
                         style="border-radius: 50%; object-fit: cover; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                </div>
                """,
        unsafe_allow_html=True,
    )
        
    with col_answer:
        st.markdown("### ✨ DocuMind AI")
        st.markdown(item["answer"])

    

        st.markdown("**Sources**")
        if item["sources"]:
            for source in item["sources"]:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <b>File:</b> {source['file']}<br>
                        <b>Page:</b> {source['page']}<br>
                        <b>Chunk start index:</b> {source['start_index']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No source references were returned.")

        with st.expander("View question"):
            st.write(item["question"])
        st.markdown("---")
