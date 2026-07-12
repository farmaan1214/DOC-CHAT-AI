"""
RAG Chat App — Cloud Deployment Version
==========================================
Same pipeline as the local Ollama version, adapted to run on free hosting
(Streamlit Community Cloud) where a local Ollama server isn't available.

Pipeline:
PDF -> PyPDFLoader -> Recursive Text Splitter -> HuggingFace Embeddings
     -> ChromaDB -> MMR Retriever -> Context + Query Prompt -> Groq Llama 3 -> Answer

Swaps vs. the local version:
    - Embeddings:  Ollama nomic-embed-text  ->  HuggingFace all-MiniLM-L6-v2 (runs in-app, free, no key)
    - LLM:         Ollama llama3 (local)    ->  Groq llama3-8b-8192 (hosted, free tier, needs API key)

Get a free Groq API key: https://console.groq.com/keys

Local run:
    export GROQ_API_KEY="your_key_here"     # Windows: set GROQ_API_KEY=your_key_here
    streamlit run rag_app_cloud.py

Streamlit Community Cloud:
    Add GROQ_API_KEY under your app's Settings -> Secrets, formatted as:
    GROQ_API_KEY = "your_key_here"
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# ─────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="📄",
    layout="wide",
)
# ─────────────────────────────────────────────
# Groq API key
# Works both locally and on Streamlit Community Cloud
# ─────────────────────────────────────────────


GROQ_API_KEY = ""

# 1. Try Streamlit Cloud secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# 2. Fall back to local environment variable
if not GROQ_API_KEY:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# ─────────────────────────────────────────────
#  Theme palettes — professional, muted, slate-based
# ─────────────────────────────────────────────
DARK = {
    "bg": "#0f1115",
    "bg_alt": "#161920",
    "sidebar_bg": "#13151b",
    "border": "#2a2e38",
    "text": "#e4e6eb",
    "text_muted": "#9096a3",
    "accent": "#5b7fdb",
    "user_bubble": "#1c2333",
    "bot_bubble": "#181b22",
    "success_bg": "#152420",
    "success_text": "#7bc9a0",
    "success_border": "#2c5c47",
    "wait_bg": "#1d2130",
    "wait_text": "#8b95ab",
    "wait_border": "#3a4258",
    "source_bg": "#151a24",
    "source_border": "#2a3548",
}

LIGHT = {
    "bg": "#fafafa",
    "bg_alt": "#ffffff",
    "sidebar_bg": "#f2f3f5",
    "border": "#e0e1e6",
    "text": "#1f2328",
    "text_muted": "#5c6370",
    "accent": "#3a5bb8",
    "user_bubble": "#eaeefb",
    "bot_bubble": "#ffffff",
    "success_bg": "#e9f6ef",
    "success_text": "#227a53",
    "success_border": "#a8dcc0",
    "wait_bg": "#eef0f4",
    "wait_text": "#5c6370",
    "wait_border": "#c9cdd6",
    "source_bg": "#f4f6fa",
    "source_border": "#d7deea",
}

T = DARK if st.session_state.dark_mode else LIGHT

# ─────────────────────────────────────────────
#  CSS — built from the active theme
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

[data-testid="stAppViewContainer"], .main {{ background: {T['bg']}; }}

[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']};
    border-right: 1px solid {T['border']};
}}
[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {T['text']} !important;
}}

.app-header {{
    background: {T['bg_alt']};
    border: 1px solid {T['border']};
    border-left: 3px solid {T['accent']};
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 22px;
}}
.app-header h1 {{ color: {T['text']}; margin: 0; font-size: 1.4rem; font-weight: 700; }}
.app-header p  {{ color: {T['text_muted']}; margin: 4px 0 0; font-size: 0.85rem; }}

.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 600; margin-top: 6px;
}}
.badge-ready {{ background: {T['success_bg']}; color: {T['success_text']}; border: 1px solid {T['success_border']}; }}
.badge-wait  {{ background: {T['wait_bg']}; color: {T['wait_text']}; border: 1px solid {T['wait_border']}; }}

.msg-user {{
    background: {T['user_bubble']};
    border: 1px solid {T['border']};
    border-radius: 10px 10px 2px 10px;
    padding: 12px 16px; margin: 8px 0 8px 50px;
    color: {T['text']}; line-height: 1.6;
}}
.msg-bot {{
    background: {T['bot_bubble']};
    border: 1px solid {T['border']};
    border-radius: 10px 10px 10px 2px;
    padding: 12px 16px; margin: 8px 50px 8px 0;
    color: {T['text']}; line-height: 1.6;
}}
.msg-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 5px; }}
.label-user {{ color: {T['accent']}; }}
.label-bot  {{ color: {T['text_muted']}; }}

.source-card {{
    background: {T['source_bg']};
    border: 1px solid {T['source_border']};
    border-left: 3px solid {T['accent']};
    border-radius: 6px;
    padding: 8px 12px; margin-top: 6px;
    font-size: 0.8rem; color: {T['text_muted']};
}}
.source-card strong {{ color: {T['text']}; }}

[data-testid="stFileUploader"] {{
    background: {T['bg_alt']};
    border: 1.5px dashed {T['border']};
    border-radius: 8px;
    padding: 8px;
}}

[data-testid="stChatInput"] textarea {{
    background: {T['bg_alt']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
}}

.stButton > button {{
    background: {T['accent']};
    color: white; border: none; border-radius: 6px;
    font-weight: 600; padding: 8px 18px;
}}
.stButton > button:hover {{ opacity: 0.88; }}

hr {{ border-color: {T['border']}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(pdf_bytes: bytes, file_name: str):
    """Load PDF → chunk → embed → ChromaDB (runs once per upload)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_chat",
    )
    return vector_store, len(documents), len(chunks)


def build_prompt(history: list, context: str, question: str) -> str:
    """Combine conversation history + retrieved context + new question."""
    memory_text = ""
    if history:
        recent = history[-4:]          # last 4 messages = last 2 turns
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        memory_text = "\n".join(lines)

    prompt = f"""You are a helpful AI assistant. Answer the user's question using the document context below.
If the answer is not in the context, say "I couldn't find that in the document."
Keep the answer clear and concise.

{'Conversation so far:' if memory_text else ''}
{memory_text}

Relevant document context:
{context}

Current question: {question}

Answer:"""
    return prompt


def ask(question: str) -> dict:
    """MMR retrieve → build prompt with memory → generate answer."""
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(st.session_state.messages, context, question)

    llm = load_llm()
    response = llm.invoke(prompt)

    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:120].replace("\n", " "),
        }
        for doc in docs
    ]
    return {"answer": response.content, "sources": sources}


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 DocChat AI")
    st.markdown("Upload a PDF and ask questions. The AI remembers your conversation.")

    if not GROQ_API_KEY:
        st.error("No GROQ_API_KEY found. Add it in Settings → Secrets (cloud) or as an environment variable (local).")

    mode_label = "🌙 Dark mode" if st.session_state.dark_mode else "☀️ Light mode"
    toggled = st.toggle(mode_label, value=st.session_state.dark_mode)
    if toggled != st.session_state.dark_mode:
        st.session_state.dark_mode = toggled
        st.rerun()

    st.markdown("---")

    st.markdown("### Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Reading and indexing PDF…"):
                vs, pages, chunks = build_vector_store(uploaded_file.read(), uploaded_file.name)
                st.session_state.vector_store = vs
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

            st.success(f"Indexed — {pages} pages, {chunks} chunks")

        st.markdown(f'<span class="badge badge-ready">📗 {uploaded_file.name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-wait">Waiting for a PDF…</span>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    steps = [
        ("PyPDFLoader", "loads your PDF"),
        ("Text splitter", "chunks into 1000-char pieces"),
        ("MiniLM embeddings", "creates vector embeddings"),
        ("ChromaDB", "stores and searches vectors"),
        ("MMR retriever", "finds relevant + diverse chunks"),
        ("Memory", "remembers the last 2 turns"),
        ("Llama 3 (Groq)", "generates the final answer"),
    ]
    for name, desc in steps:
        st.markdown(f"**{name}** — {desc}")


# ─────────────────────────────────────────────
#  Main area — header
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <h1>DocChat AI</h1>
    <p>Ask anything about your PDF — the AI reads it, remembers your conversation, and answers from the document.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Chat history display
# ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(f"""
    <div style="text-align:center; padding: 60px 20px; color: {T['text_muted']};">
        <div style="font-size:1.05rem; font-weight:600; color:{T['text']};">
            Upload a PDF from the sidebar to get started
        </div>
        <div style="font-size:0.85rem; margin-top:8px;">
            Then ask any question — the AI will answer from the document and remember the conversation.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label label-user">You</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-bot">
            <div class="msg-label label-bot">DocChat AI</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} source chunks used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Chunk {i} · Page {src['page']}</strong><br>
                        "{src['snippet']}…"
                    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Chat input
# ─────────────────────────────────────────────
if st.session_state.vector_store is None:
    st.chat_input("Upload a PDF first…", disabled=True)
elif not GROQ_API_KEY:
    st.chat_input("Add your GROQ_API_KEY to enable chat…", disabled=True)
else:
    question = st.chat_input("Ask a question about your PDF…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking…"):
            result = ask(question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()
