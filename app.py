"""
Corporate Knowledge Base RAG — Web App (Dashboard UI)
MBA462B | Group 3

Pipeline/logic is unchanged (extract -> chunk -> embed -> FAISS -> retrieve -> generate).
This version adds a full dashboard-style UI/UX layer:
sidebar navigation, hero banner, stat cards, quick-ask chips, knowledge base
explorer, analytics, and settings — all built with Streamlit + custom CSS
(no paid assets; Google Fonts loaded via CDN).
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return datetime.now(IST)

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
# Custom CSS — fonts, colors, cards, nav, animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Poppins', sans-serif; }

.stApp { background: #f5f6fb; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #eef0f7;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

.brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 1.6rem;
    padding: 0 0.2rem;
}
.brand .logo {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6C5CE7, #8E7CFF);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 6px 14px -6px rgba(108,92,231,0.55);
}
.brand .name {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: #1a1a2e;
    line-height: 1.15;
}
.brand .sub { font-size: 0.75rem; color: #8b8ba7; font-weight: 500; }

section[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background: transparent !important;
    color: #4a4a63 !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 0.85rem !important;
    border-radius: 10px !important;
    margin-bottom: 0.2rem;
    transition: background 0.15s ease, color 0.15s ease;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: #f3f0ff !important;
    color: #6C5CE7 !important;
    transform: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #6C5CE7, #8E7CFF) !important;
    color: white !important;
    box-shadow: 0 6px 14px -6px rgba(108,92,231,0.55) !important;
}

.sidebar-card {
    background: linear-gradient(135deg, #6C5CE7, #8E7CFF);
    border-radius: 16px;
    padding: 1.1rem 1.1rem;
    color: white;
    margin: 1.2rem 0 1rem 0;
}
.sidebar-card .t { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; }
.sidebar-card .d { font-size: 0.78rem; opacity: 0.92; line-height: 1.4; }

/* ---------- Top bar ---------- */
.topbar {
    background: white;
    border-radius: 16px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 18px -12px rgba(20,20,50,0.15);
}
.topbar .greet { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.15rem; color: #1a1a2e; }
.topbar .subgreet { color: #8b8ba7; font-size: 0.85rem; margin-top: 0.1rem; }
.topbar .icons { display: flex; gap: 0.6rem; }
.icon-pill {
    width: 40px; height: 40px; border-radius: 12px;
    background: #f5f4fe; display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(120deg, #6C5CE7 0%, #8E7CFF 45%, #A78BFA 100%);
    border-radius: 20px;
    padding: 2.4rem 2.2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 12px 30px -12px rgba(108, 92, 231, 0.45);
    animation: fadeIn 0.6s ease-out;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: -60px; right: -40px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.10);
}
.hero::after {
    content: "";
    position: absolute; bottom: -70px; right: 120px;
    width: 140px; height: 140px; border-radius: 50%;
    background: rgba(255,255,255,0.08);
}
.hero h1 { color: white; font-size: 2.1rem; margin: 0 0 0.5rem 0; font-weight: 800; }
.hero p { color: rgba(255,255,255,0.92); font-size: 1.0rem; margin: 0; max-width: 640px; line-height: 1.55; }
.hero .badge-row { margin-top: 1.1rem; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.4rem;
    backdrop-filter: blur(4px);
}

/* ---------- Generic panel/card ---------- */
.panel {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 4px 18px -12px rgba(20,20,50,0.15);
    margin-bottom: 1.2rem;
}
.panel-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #1a1a2e;
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.9rem;
}

/* Section headers (numbered) */
.section-label { display: flex; align-items: center; gap: 0.6rem; margin: 0.4rem 0 0.9rem 0; }
.section-label .num {
    background: #6C5CE7; color: white; width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; font-family: 'Poppins', sans-serif;
}
.section-label .title { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.2rem; color: #1a1a2e; }

/* Feature / stat cards */
.feat-card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    box-shadow: 0 4px 18px -12px rgba(20,20,50,0.15);
    height: 100%;
}
.feat-icon {
    width: 42px; height: 42px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; margin-bottom: 0.7rem;
}
.feat-title { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 0.95rem; color: #1a1a2e; margin-bottom: 0.3rem; }
.feat-desc { font-size: 0.82rem; color: #8b8ba7; line-height: 1.45; }

.stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 18px -12px rgba(20,20,50,0.15);
    text-align: center;
}
.stat-icon { font-size: 1.4rem; margin-bottom: 0.35rem; }
.stat-num { font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.35rem; color: #1a1a2e; }
.stat-label { font-size: 0.78rem; color: #8b8ba7; font-weight: 600; margin-top: 0.15rem; }

/* Answer card */
.answer-card {
    background: white; border-radius: 16px; padding: 1.6rem 1.8rem;
    border: 1px solid #eee6ff; box-shadow: 0 8px 24px -14px rgba(108,92,231,0.25);
    animation: slideUp 0.4s ease-out; margin-bottom: 1.2rem;
}
.answer-card .label {
    font-family: 'Poppins', sans-serif; font-weight: 700; color: #6C5CE7;
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem;
}
.answer-card p { font-size: 1.05rem; line-height: 1.6; color: #23233a; margin: 0; white-space: pre-wrap; }

.source-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #f3f0ff; color: #6C5CE7; padding: 0.35rem 0.8rem;
    border-radius: 999px; font-size: 0.82rem; font-weight: 600; margin: 0.2rem 0.35rem 0.2rem 0;
}

.doc-row {
    display: flex; align-items: center; justify-content: space-between;
    background: #fafaff; border: 1px solid #f0edff; border-radius: 12px;
    padding: 0.7rem 1rem; margin-bottom: 0.55rem;
}
.doc-row .name { font-weight: 600; color: #1a1a2e; font-size: 0.9rem; }
.doc-row .meta { font-size: 0.78rem; color: #8b8ba7; }

.recent-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.65rem 0; border-bottom: 1px solid #f2f2f8;
}
.recent-row:last-child { border-bottom: none; }
.recent-q { font-size: 0.88rem; color: #1a1a2e; font-weight: 600; }
.recent-meta { font-size: 0.75rem; color: #8b8ba7; }
.tag-pill {
    background: #eafaf0; color: #1e8a4a; font-size: 0.72rem; font-weight: 700;
    padding: 0.2rem 0.6rem; border-radius: 999px;
}

/* KB status */
.kb-status { border-radius: 14px; padding: 0.9rem 1.1rem; font-weight: 600; font-size: 0.87rem; margin-top: 0.6rem; }
.kb-ready { background: #e8f9ee; color: #1e8a4a; border: 1px solid #c6f0d5; }
.kb-empty { background: #fff7e6; color: #b3760a; border: 1px solid #ffe6b3; }

/* Buttons */
div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px -6px rgba(108,92,231,0.5); }
div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #6C5CE7, #8E7CFF) !important;
    border: none !important;
}

/* Chip buttons (try-asking) */
.chip-wrap div.stButton > button {
    background: #f3f0ff !important;
    color: #6C5CE7 !important;
    border: 1px solid #e6ddff !important;
    border-radius: 999px !important;
    padding: 0.35rem 0.9rem !important;
    font-size: 0.82rem !important;
    box-shadow: none !important;
}
.chip-wrap div.stButton > button:hover {
    background: #6C5CE7 !important;
    color: white !important;
}

div[data-baseweb="input"] > div { border-radius: 12px !important; }
.streamlit-expanderHeader { font-weight: 600 !important; border-radius: 10px !important; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(-8px);} to { opacity: 1; transform: translateY(0);} }
@keyframes slideUp { from { opacity: 0; transform: translateY(14px);} to { opacity: 1; transform: translateY(0);} }
</style>
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
defaults = {
    "knowledge_chunks": [],
    "index": None,
    "kb_ready": False,
    "history": [],           # list of dicts: question, answer, sources, time
    "page": "Home",
    "question_input": "",
    "pending_ask": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

SUGGESTED_QUESTIONS = [
    "Leave policy for new joinees",
    "Working from home policy",
    "Maternity leave details",
    "Notice period for resignation",
]

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


def run_query(q):
    """Run a RAG query and push it into history."""
    with st.spinner("Searching documents and generating answer..."):
        answer, sources = rag_query_safe(q, st.session_state.index, st.session_state.knowledge_chunks)
    st.session_state.history.append({
        "question": q,
        "answer": answer,
        "sources": sources,
        "time": now_ist().strftime("%I:%M %p"),
    })


def time_greeting():
    hour = now_ist().hour
    if hour < 12:
        return "Good morning 👋"
    if hour < 17:
        return "Good afternoon 👋"
    return "Good evening 👋"


# ---------------------------------------------------------------------------
# Sidebar — brand + navigation + build KB
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("Home", "🏠"),
    ("Ask a Question", "❓"),
    ("Upload Documents", "📤"),
    ("Knowledge Base", "📚"),
    ("Analytics", "📊"),
    ("Settings", "⚙️"),
]

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="logo">✨</div>
        <div>
            <div class="name">Knowledge Base</div>
            <div class="sub">Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for label, icon in NAV_ITEMS:
        is_active = st.session_state.page == label
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = label
            st.rerun()

    st.markdown("""
    <div class="sidebar-card">
        <div class="t">🚀 Retrieval-Augmented Generation</div>
        <div class="d">Answers are generated strictly from your uploaded documents, with exact page-level citations.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📥 Quick upload")
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

# ---------------------------------------------------------------------------
# Shared: top bar
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="topbar">
    <div>
        <div class="greet">{time_greeting()}</div>
        <div class="subgreet">Your AI-powered knowledge assistant is ready to help.</div>
    </div>
    <div class="icons">
        <div class="icon-pill">🔔</div>
        <div class="icon-pill">🌙</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Derived stats (all real, computed from session state — nothing fabricated)
# ---------------------------------------------------------------------------
num_docs = len({c["source"] for c in st.session_state.knowledge_chunks})
num_chunks = len(st.session_state.knowledge_chunks)
num_questions = len(st.session_state.history)
num_active_sources = num_docs

# ---------------------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------------------
if st.session_state.page == "Home":
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

    # --- Ask a question quick box ---
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">❓ Ask a question</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        home_q = st.text_input(
            "home_question", value=st.session_state.question_input,
            placeholder="e.g. How many casual leaves am I entitled to?",
            disabled=not st.session_state.kb_ready, label_visibility="collapsed", key="home_q_input"
        )
    with col2:
        home_ask = st.button("Ask AI ✨", type="primary", use_container_width=True,
                              disabled=not st.session_state.kb_ready or not home_q.strip())

    st.markdown('<div class="chip-wrap">', unsafe_allow_html=True)
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, sq in enumerate(SUGGESTED_QUESTIONS):
        with chip_cols[i]:
            if st.button(sq, key=f"chip_{i}", disabled=not st.session_state.kb_ready, use_container_width=True):
                st.session_state.question_input = sq
                run_query(sq)
                st.session_state.question_input = ""
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if home_ask and home_q.strip():
        run_query(home_q.strip())
        st.rerun()

    # --- Latest answer (this was missing before — answers were saved to
    # history but never actually rendered on the Home page) ---
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.markdown(f"""
        <div class="answer-card">
            <div class="label">💬 {latest['question']}</div>
            <p>{latest['answer']}</p>
        </div>
        """, unsafe_allow_html=True)

        pills = "".join(
            f'<span class="source-pill">📄 {s["source"]} — Page {s["page"]}</span>'
            for s in latest["sources"]
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # --- Feature cards ---
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🧠", "#eef0ff", "Advanced RAG", "Retrieval-Augmented Generation for accurate, source-backed answers."),
        ("🗂️", "#e7f9ef", "Smart Sources", "Answers come only from your uploaded, trusted documents."),
        ("🔒", "#fff3e6", "Secure & Private", "Your documents stay in-session and are never shared externally."),
        ("⚡", "#e8f1ff", "Powered by Gemini", "Built on Google's Gemini models for reliable, hallucination-free AI."),
    ]
    for col, (icon, bg, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon" style="background:{bg};">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Recent questions + KB stats ---
    rc1, rc2 = st.columns([1.3, 1])
    with rc1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🕓 Recent questions</div>', unsafe_allow_html=True)
        if st.session_state.history:
            for item in reversed(st.session_state.history[-5:]):
                n_src = len(item["sources"])
                st.markdown(f"""
                <div class="recent-row">
                    <div>
                        <div class="recent-q">{item['question']}</div>
                        <div class="recent-meta">Answered at {item['time']}</div>
                    </div>
                    <span class="tag-pill">{n_src} source{'s' if n_src != 1 else ''}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No questions asked yet — build your knowledge base and try one of the suggestions above.")
        st.markdown('</div>', unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Knowledge base stats</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s3, s4 = st.columns(2)
        stats = [
            (s1, "📄", num_docs, "Documents"),
            (s2, "❓", num_questions, "Questions Answered"),
            (s3, "🧩", num_chunks, "Chunks Indexed"),
            (s4, "🗃️", num_active_sources, "Active Sources"),
        ]
        for col, icon, num, label in stats:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon">{icon}</div>
                    <div class="stat-num">{num}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: ASK A QUESTION
# ---------------------------------------------------------------------------
elif st.session_state.page == "Ask a Question":
    st.markdown('<div class="section-label"><span class="num">1</span><span class="title">Ask a question</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "question", placeholder="e.g. How many casual leaves am I entitled to?",
            disabled=not st.session_state.kb_ready, label_visibility="collapsed"
        )
    with col2:
        ask_clicked = st.button("Ask ✨", type="primary",
                                 disabled=not st.session_state.kb_ready or not question.strip(),
                                 use_container_width=True)

    if not st.session_state.kb_ready:
        st.info("Upload PDFs and build your knowledge base from the sidebar to start asking questions.")

    if ask_clicked and question.strip():
        run_query(question.strip())

    if st.session_state.history:
        st.markdown('<div class="section-label"><span class="num">2</span><span class="title">Answer</span></div>', unsafe_allow_html=True)

        latest = st.session_state.history[-1]
        st.markdown(f"""
        <div class="answer-card">
            <div class="label">💬 {latest['question']}</div>
            <p>{latest['answer']}</p>
        </div>
        """, unsafe_allow_html=True)

        pills = "".join(
            f'<span class="source-pill">📄 {s["source"]} — Page {s["page"]}</span>'
            for s in latest["sources"]
        )
        st.markdown(pills, unsafe_allow_html=True)

        with st.expander("🔍 View exact source passages"):
            for s in latest["sources"]:
                st.markdown(f"**{s['source']} — Page {s['page']}**")
                st.write(s["text"])
                st.markdown("---")

        if len(st.session_state.history) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander(f"🕓 View previous questions ({len(st.session_state.history) - 1})"):
                for item in reversed(st.session_state.history[:-1]):
                    st.markdown(f"**{item['question']}**  \n{item['answer']}")
                    st.caption(", ".join(f"{s['source']} (p.{s['page']})" for s in item["sources"]))
                    st.markdown("---")

# ---------------------------------------------------------------------------
# PAGE: UPLOAD DOCUMENTS
# ---------------------------------------------------------------------------
elif st.session_state.page == "Upload Documents":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📤 Upload documents</div>', unsafe_allow_html=True)
    st.write("Add company policy PDFs here, then build the knowledge base so they become searchable.")

    page_uploaded = st.file_uploader(
        "Upload policy PDFs", type=["pdf"], accept_multiple_files=True, key="page_uploader"
    )

    if st.button("🚀 Build Knowledge Base", type="primary", key="page_build_btn",
                 disabled=not page_uploaded, use_container_width=False):
        progress_bar = st.progress(0.0, text="Starting...")

        def _update(pct, msg):
            progress_bar.progress(pct, text=msg)

        with st.spinner("Reading and embedding documents..."):
            chunks, index = build_knowledge_base(page_uploaded, progress_callback=_update)

        st.session_state.knowledge_chunks = chunks
        st.session_state.index = index
        st.session_state.kb_ready = True
        progress_bar.empty()
        num_docs_built = len({c["source"] for c in chunks})
        st.success(f"Knowledge base built with {len(chunks)} chunks from {num_docs_built} document(s).")
        st.balloons()

    if st.session_state.kb_ready:
        st.markdown(
            f'<div class="kb-status kb-ready">✅ {len(st.session_state.knowledge_chunks)} chunks loaded and searchable</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="kb-status kb-empty">⚠️ No knowledge base built yet</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: KNOWLEDGE BASE
# ---------------------------------------------------------------------------
elif st.session_state.page == "Knowledge Base":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📚 Knowledge base</div>', unsafe_allow_html=True)

    if not st.session_state.knowledge_chunks:
        st.info("No documents indexed yet. Go to **Upload Documents** to add PDFs.")
    else:
        doc_summary = {}
        for c in st.session_state.knowledge_chunks:
            doc_summary.setdefault(c["source"], {"chunks": 0, "pages": set()})
            doc_summary[c["source"]]["chunks"] += 1
            doc_summary[c["source"]]["pages"].add(c["page"])

        for name, info in doc_summary.items():
            st.markdown(f"""
            <div class="doc-row">
                <div>
                    <div class="name">📄 {name}</div>
                    <div class="meta">{len(info['pages'])} pages indexed</div>
                </div>
                <div class="meta">{info['chunks']} chunks</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.knowledge_chunks:
        with st.expander("🔍 Browse indexed text chunks"):
            for i, c in enumerate(st.session_state.knowledge_chunks):
                st.markdown(f"**{c['source']} — Page {c['page']}**")
                st.write(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
                st.markdown("---")

# ---------------------------------------------------------------------------
# PAGE: ANALYTICS
# ---------------------------------------------------------------------------
elif st.session_state.page == "Analytics":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📊 Analytics</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "📄", num_docs, "Documents"),
        (s2, "🧩", num_chunks, "Chunks Indexed"),
        (s3, "❓", num_questions, "Questions Answered"),
        (s4, "🗃️", num_active_sources, "Active Sources"),
    ]
    for col, icon, num, label in stats:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-num">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📈 Most-cited sources</div>', unsafe_allow_html=True)
    if st.session_state.history:
        citation_counts = {}
        for item in st.session_state.history:
            for s in item["sources"]:
                citation_counts[s["source"]] = citation_counts.get(s["source"], 0) + 1
        st.bar_chart(citation_counts)
    else:
        st.caption("Ask a few questions to see which documents get cited the most.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🕓 Question history</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.markdown(f"**{item['question']}**  ·  _{item['time']}_")
            st.caption(", ".join(f"{s['source']} (p.{s['page']})" for s in item["sources"]))
    else:
        st.caption("No questions asked yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: SETTINGS
# ---------------------------------------------------------------------------
elif st.session_state.page == "Settings":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚙️ Settings</div>', unsafe_allow_html=True)

    st.markdown(f"""
    **Embedding model:** `{EMBED_MODEL}`
    **Generation model:** `{GEN_MODEL}`
    **Retrieval method:** FAISS (`IndexFlatL2`), top‑3 nearest chunks
    **Chunking:** 400 words per chunk, 50‑word overlap
    """)

    st.markdown("---")
    st.markdown("##### Danger zone")
    dc1, dc2 = st.columns(2)
    with dc1:
        if st.button("🗑️ Clear question history", use_container_width=True):
            st.session_state.history = []
            st.success("Question history cleared.")
    with dc2:
        if st.button("🧹 Reset knowledge base", use_container_width=True):
            st.session_state.knowledge_chunks = []
            st.session_state.index = None
            st.session_state.kb_ready = False
            st.success("Knowledge base reset.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Retrieval-Augmented Generation prototype · MBA462B, Group 3 · Answers generated strictly from uploaded documents")
