"""
Corporate Knowledge Base RAG — Web App (Redesigned UI)
MBA462B — CIA-3 | Group 3

Same pipeline and logic as before (extract -> chunk -> embed -> FAISS -> retrieve -> generate).
This version adds a fully custom visual design layer using CSS injection:
gradient hero, Google Fonts, styled cards, chat-style answers, animated states.
"""

import time
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Knowledge Base Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — fonts, colors, cards, animations
# All free: Google Fonts (Poppins + Inter) loaded via CDN, no paid assets used.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 { font-family: 'Poppins', sans-serif; }

/* App background */
.stApp {
    background: linear-gradient(180deg, #f6f7fb 0%, #ffffff 35%);
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #6C5CE7 0%, #8E7CFF 45%, #A78BFA 100%);
    border-radius: 20px;
    padding: 2.6rem 2.2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 12px 30px -12px rgba(108, 92, 231, 0.45);
    animation: fadeIn 0.6s ease-out;
}
.hero h1 {
    color: white;
    font-size: 2.3rem;
    margin: 0 0 0.5rem 0;
    font-weight: 800;
}
.hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1.02rem;
    margin: 0;
    max-width: 640px;
}
.hero .badge-row { margin-top: 1rem; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-right: 0.5rem;
    backdrop-filter: blur(4px);
}

/* Section headers */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.4rem 0 0.9rem 0;
}
.section-label .num {
    background: #6C5CE7;
    color: white;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
    font-family: 'Poppins', sans-serif;
}
.section-label .title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    color: #1a1a2e;
}

/* Answer card */
.answer-card {
    background: white;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    border: 1px solid #eee6ff;
    box-shadow: 0 8px 24px -14px rgba(108,92,231,0.25);
    animation: slideUp 0.4s ease-out;
    margin-bottom: 1.2rem;
}
.answer-card .label {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #6C5CE7;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
}
.answer-card p {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #23233a;
    margin: 0;
}

/* Source pill cards */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f3f0ff;
    color: #6C5CE7;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.2rem 0.35rem 0.2rem 0;
}

/* KB status card */
.kb-status {
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    font-weight: 600;
    font-size: 0.9rem;
    margin-top: 0.8rem;
}
.kb-ready { background: #e8f9ee; color: #1e8a4a; border: 1px solid #c6f0d5; }
.kb-empty { background: #fff7e6; color: #b3760a; border: 1px solid #ffe6b3; }

/* Buttons */
div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px -6px rgba(108,92,231,0.5);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #6C5CE7, #8E7CFF) !important;
    border: none !important;
}

/* Text input */
div[data-baseweb="input"] > div {
    border-radius: 12px !important;
}

/* Expander (source snippets) */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    border-radius: 10px !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>✨ Corporate Knowledge Base Assistant</h1>
    <p>Upload company policy PDFs and ask questions in plain language.
    Every answer is grounded strictly in your documents, with exact source citations — zero hallucination.</p>
    <div class="badge-row">
        <span class="badge">🔍 Retrieval-Augmented Generation</span>
        <span class="badge">📎 Source-cited answers</span>
        <span class="badge">⚡ Powered by Gemini</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Connect to Gemini
# ---------------------------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Settings → Secrets, then reload.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash-lite"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "knowledge_chunks" not in st.session_state:
    st.session_state.knowledge_chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, sources)

# ---------------------------------------------------------------------------
# Core pipeline functions (unchanged logic)
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values


def build_knowledge_base(uploaded_files, progress_callback=None):
    knowledge_chunks = []
    for f in uploaded_files:
        pages = extract_text_from_pdf(f)
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                knowledge_chunks.append({"source": f.name, "page": page_num, "text": chunk})

    embeddings = []
    total = len(knowledge_chunks)
    for i, c in enumerate(knowledge_chunks):
        embeddings.append(embed_text(c["text"]))
        if progress_callback:
            progress_callback((i + 1) / total, f"Embedding chunk {i + 1}/{total}...")
        if i % 20 == 0 and i > 0:
            time.sleep(1)

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return knowledge_chunks, index


def retrieve_top_chunks(query, index, knowledge_chunks, top_k=3):
    query_vec = embed_text(query, task_type="RETRIEVAL_QUERY")
    query_vec = np.array([query_vec]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [knowledge_chunks[i] for i in indices[0]]


def generate_answer(query, retrieved_chunks):
    context_text = "\n\n".join(
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    prompt = f"""You are a corporate knowledge base assistant.
Answer the employee's question ONLY using the context below.
If the answer isn't in the context, say you don't have that information -- do not guess.
Always cite the source filename and page number.

Context:
{context_text}

Question: {query}

Answer (concise, plain language, with citation):"""

    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text


def rag_query_safe(query, index, knowledge_chunks, top_k=3, max_retries=3):
    for attempt in range(max_retries):
        try:
            top_chunks = retrieve_top_chunks(query, index, knowledge_chunks, top_k)
            answer = generate_answer(query, top_chunks)
            return answer, top_chunks
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                st.toast(f"Rate limit hit, retrying in 20s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(20)
            else:
                raise
    raise RuntimeError("Failed after max retries due to rate limiting.")


# ---------------------------------------------------------------------------
# Sidebar — build knowledge base
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📥 Build Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload policy PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    if st.button("🚀 Build Knowledge Base", type="primary", disabled=not uploaded_files, use_container_width=True):
        progress_bar = st.progress(0.0, text="Starting...")

        def _update(pct, msg):
            progress_bar.progress(pct, text=msg)

        with st.spinner("Reading and embedding documents..."):
            chunks, index = build_knowledge_base(uploaded_files, progress_callback=_update)

        st.session_state.knowledge_chunks = chunks
        st.session_state.index = index
        st.session_state.kb_ready = True
        progress_bar.empty()
        st.balloons()

    if st.session_state.kb_ready:
        st.markdown(
            f'<div class="kb-status kb-ready">✅ {len(st.session_state.knowledge_chunks)} chunks loaded and searchable</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="kb-status kb-empty">⚠️ Upload PDFs and build the knowledge base to start</div>',
            unsafe_allow_html=True
        )

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕓 Recent questions")
        for q, _, _ in reversed(st.session_state.history[-5:]):
            st.caption(f"• {q}")

# ---------------------------------------------------------------------------
# Main area — ask a question
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label"><span class="num">1</span><span class="title">Ask a question</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "question", placeholder="e.g. How many casual leaves am I entitled to?",
        disabled=not st.session_state.kb_ready, label_visibility="collapsed"
    )
with col2:
    ask_clicked = st.button("Ask ✨", type="primary", disabled=not st.session_state.kb_ready or not question.strip(), use_container_width=True)

if ask_clicked:
    with st.spinner("Searching documents and generating answer..."):
        answer, sources = rag_query_safe(
            question, st.session_state.index, st.session_state.knowledge_chunks
        )
    st.session_state.history.append((question, answer, sources))

if st.session_state.history:
    st.markdown('<div class="section-label"><span class="num">2</span><span class="title">Answer</span></div>', unsafe_allow_html=True)

    latest_q, latest_a, latest_sources = st.session_state.history[-1]
    st.markdown(f"""
    <div class="answer-card">
        <div class="label">💬 {latest_q}</div>
        <p>{latest_a}</p>
    </div>
    """, unsafe_allow_html=True)

    pills = "".join(
        f'<span class="source-pill">📄 {s["source"]} — Page {s["page"]}</span>'
        for s in latest_sources
    )
    st.markdown(pills, unsafe_allow_html=True)

    with st.expander("🔍 View exact source passages"):
        for s in latest_sources:
            st.markdown(f"**{s['source']} — Page {s['page']}**")
            st.write(s["text"])
            st.markdown("---")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Retrieval-Augmented Generation prototype · MBA462B, CIA-3, Group 3 · Answers generated strictly from uploaded documents")
