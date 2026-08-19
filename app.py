"""Corporate Knowledge Base Assistant — an evidence-first Streamlit RAG workspace."""

import time
from datetime import datetime, timezone
from html import escape

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pypdf import PdfReader


st.set_page_config(
    page_title="Evidence Desk | Knowledge Base",
    page_icon="📇",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        :root { --ink:#172126; --paper:#f6f2ea; --surface:#fffdf8; --rule:#d7d0c4; --muted:#59656a; --red:#b83a32; --green:#28624f; --ochre:#9a6a24; }
        .stApp { background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans', sans-serif; }
        .stApp:before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.16; background-image:radial-gradient(#8d8274 .45px, transparent .45px); background-size:7px 7px; }
        [data-testid="stHeader"] { background:rgba(246,242,234,.94); }
        [data-testid="stSidebar"] { background:var(--ink); border-right:1px solid #2c393d; }
        [data-testid="stSidebar"] * { color:#f6f2ea; }
        [data-testid="stSidebar"] .stCaption { color:#aeb7b5; }
        h1,h2,h3 { font-family:'Cormorant Garamond', serif !important; color:var(--ink); letter-spacing:-.025em; }
        h1 { font-size:clamp(2.3rem, 5vw, 4.3rem) !important; line-height:.95 !important; margin-bottom:.3rem !important; }
        h2 { font-size:2rem !important; }
        .eyebrow { font:600 .68rem 'IBM Plex Mono', monospace; letter-spacing:.18em; text-transform:uppercase; color:var(--red); }
        .hero { padding:2.25rem 0 1.7rem; border-bottom:2px solid var(--ink); margin-bottom:1.6rem; }
        .hero-copy { max-width:750px; color:var(--muted); font-size:1.05rem; line-height:1.65; }
        .section-label { font:600 .7rem 'IBM Plex Mono', monospace; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin:1.5rem 0 .7rem; }
        .panel { background:var(--surface); border:1px solid var(--rule); padding:1.35rem; margin-bottom:1rem; box-shadow:4px 4px 0 rgba(23,33,38,.05); }
        .panel-dark { background:#202c30; color:#f6f2ea; border:1px solid #39474a; padding:1.2rem; }
        .panel-dark .eyebrow { color:#df8078; }
        .readiness { display:flex; gap:0; border:1px solid var(--rule); background:var(--surface); margin-bottom:2rem; }
        .stage { flex:1; padding:1rem 1.1rem; border-right:1px solid var(--rule); }
        .stage:last-child { border-right:0; }
        .stage-kicker { font:500 .68rem 'IBM Plex Mono', monospace; color:var(--muted); text-transform:uppercase; letter-spacing:.12em; }
        .stage-value { font-size:1.02rem; font-weight:600; margin-top:.35rem; }
        .stage-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#a7ada8; margin-right:7px; }
        .stage-dot.ready { background:var(--green); box-shadow:0 0 0 4px #dce9e2; }
        .stage-dot.warn { background:var(--ochre); }
        .answer { border-left:4px solid var(--red); background:var(--surface); border-top:1px solid var(--rule); border-right:1px solid var(--rule); border-bottom:1px solid var(--rule); padding:1.5rem; animation:appear .35s ease-out; }
        .question-quote { color:var(--muted); font-style:italic; border-left:2px solid var(--rule); padding-left:1rem; margin:.5rem 0 1.2rem; }
        .citation { display:inline-block; background:#e8ded0; color:var(--ink); padding:.4rem .65rem; margin:.25rem .25rem 0 0; font:500 .74rem 'IBM Plex Mono', monospace; }
        .source-text { background:#f1ece3; border-left:2px solid var(--red); padding:1rem; line-height:1.7; color:#334146; }
        .mono { font-family:'IBM Plex Mono', monospace; }
        [data-testid="stButton"] button { border-radius:2px; font-weight:600; min-height:44px; transition:transform .18s ease, background-color .18s ease; }
        [data-testid="stButton"] button:hover { transform:translateY(-1px); }
        [data-testid="stFileUploader"] { border:1px dashed #9f9588; background:#faf7f0; padding:.35rem; }
        textarea, input { border-radius:2px !important; background:#fffdf8 !important; }
        @keyframes appear { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }
        @media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation:none !important; transition:none !important; } }
        @media (max-width: 700px) { .readiness { flex-direction:column; } .stage { border-right:0; border-bottom:1px solid var(--rule); } .stage:last-child { border-bottom:0; } .hero { padding-top:1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
if not api_key:
    st.error("Gemini is not connected. Add GEMINI_API_KEY in Streamlit Secrets, then reload.", icon="⚠️")
    st.stop()

client = genai.Client(api_key=api_key)
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash-lite"

defaults = {"knowledge_chunks": [], "index": None, "kb_ready": False, "history": [], "documents": [], "selected_source": None}
for name, value in defaults.items():
    if name not in st.session_state:
        st.session_state[name] = value


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        value = " ".join(words[start : start + chunk_size]).strip()
        if value:
            chunks.append(value)
        start += max(1, chunk_size - overlap)
    return chunks


def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    result = client.models.embed_content(model=EMBED_MODEL, contents=text, config=types.EmbedContentConfig(task_type=task_type))
    return result.embeddings[0].values


def build_knowledge_base(uploaded_files, progress_callback=None):
    chunks = []
    for file in uploaded_files:
        for page_num, page_text in extract_text_from_pdf(file):
            for chunk in chunk_text(page_text):
                chunks.append({"source": file.name, "page": page_num, "text": chunk})
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")
    vectors = []
    for i, item in enumerate(chunks):
        vectors.append(embed_text(item["text"]))
        if progress_callback:
            progress_callback((i + 1) / len(chunks), f"Indexing evidence {i + 1} of {len(chunks)}")
        if i and i % 20 == 0:
            time.sleep(1)
    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return chunks, index


def retrieve_top_chunks(query, index, chunks, top_k=3):
    query_vector = np.asarray([embed_text(query, "RETRIEVAL_QUERY")], dtype="float32")
    _, indices = index.search(query_vector, min(top_k, len(chunks)))
    return [chunks[i] for i in indices[0] if i >= 0]


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}" for c in retrieved_chunks)
    prompt = f"""You are a corporate knowledge base assistant. Answer ONLY from the context. If the answer is absent, say you do not have that information—do not guess. Always cite filename and page number.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer in concise plain language with citations:"""
    return client.models.generate_content(model=GEN_MODEL, contents=prompt).text


def rag_query_safe(query):
    for attempt in range(3):
        try:
            sources = retrieve_top_chunks(query, st.session_state.index, st.session_state.knowledge_chunks)
            return generate_answer(query, sources), sources
        except ClientError as error:
            if "RESOURCE_EXHAUSTED" not in str(error) or attempt == 2:
                raise
            time.sleep(20)
    raise RuntimeError("The request could not be completed after three attempts.")


with st.sidebar:
    st.markdown("<div class='eyebrow'>Evidence Desk / 01</div>", unsafe_allow_html=True)
    st.markdown("## Knowledge Base")
    st.caption("A private reading room for policy answers grounded in your documents.")
    st.markdown("<div class='section-label'>Workflow</div>", unsafe_allow_html=True)
    st.markdown("1. Load policy PDFs\n\n2. Build the evidence index\n\n3. Ask a focused question\n\n4. Inspect the passages")
    st.markdown("<div class='section-label'>Current inventory</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='panel-dark'><div class='eyebrow'>Searchable chunks</div><div style='font-size:2rem;font-weight:600'>{len(st.session_state.knowledge_chunks):,}</div><div class='mono' style='color:#aeb7b5;font-size:.7rem'>SOURCE DOCUMENTS: {len(st.session_state.documents)}</div></div>", unsafe_allow_html=True)
    if st.button("Reset workspace", key="reset-knowledge-base-button", use_container_width=True):
        for key, value in defaults.items():
            st.session_state[key] = value.copy() if isinstance(value, list) else value
        st.rerun()
    if st.session_state.history:
        st.markdown("<div class='section-label'>Recent questions</div>", unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            history_index = len(st.session_state.history) - 1 - i
            if st.button(f"{i + 1:02d}  {item['question'][:50]}", key=f"question-history-row-{history_index}", use_container_width=True):
                st.session_state["question-input"] = item["question"]
                st.rerun()


st.markdown("<div class='hero'><div class='eyebrow'>Corporate policy intelligence / grounded retrieval</div><h1>Find the answer.<br><em>Keep the evidence.</em></h1><div class='hero-copy'>Upload internal policy documents, build a searchable index, and ask questions in plain language. Every response stays tethered to an exact source passage.</div></div>", unsafe_allow_html=True)

ready = st.session_state.kb_ready
st.markdown(f"""<div class='readiness' data-testid='knowledge-readiness'>
<div class='stage'><div class='stage-kicker'>01 / Documents</div><div class='stage-value'><span class='stage-dot {'ready' if st.session_state.documents else 'warn'}'></span>{len(st.session_state.documents)} loaded</div></div>
<div class='stage'><div class='stage-kicker'>02 / Index</div><div class='stage-value'><span class='stage-dot {'ready' if ready else ''}'></span>{'Ready to search' if ready else 'Waiting to build'}</div></div>
<div class='stage'><div class='stage-kicker'>03 / Answers</div><div class='stage-value'><span class='stage-dot {'ready' if ready else ''}'></span>{'Grounded responses' if ready else 'Locked until ready'}</div></div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.markdown("<div class='section-label'>Prepare the desk</div>", unsafe_allow_html=True)
    st.markdown("### Bring in your source material")
    st.caption("PDF files only · text-based policies work best · source pages are preserved")
    uploaded_files = st.file_uploader("Upload policy PDFs", type=["pdf"], accept_multiple_files=True, key="pdf-uploader", label_visibility="collapsed")
    if uploaded_files:
        st.info(f"{len(uploaded_files)} document{'s' if len(uploaded_files) != 1 else ''} ready to index.", icon="📄")
        for file in uploaded_files:
            size_mb = (getattr(file, "size", 0) or 0) / (1024 * 1024)
            st.caption(f"{file.name} · {size_mb:.2f} MB · pages verified during indexing")
    build = st.button("Build knowledge base", type="primary", disabled=not uploaded_files, key="build-knowledge-base-button", use_container_width=True)
    if build:
        st.session_state.knowledge_chunks = []
        st.session_state.index = None
        st.session_state.documents = []
        st.session_state.kb_ready = False
        st.session_state.selected_source = None
        st.markdown("<div data-testid='embedding-progress'></div>", unsafe_allow_html=True)
        progress = st.progress(0, text="Reading documents…")
        try:
            with st.status("Preparing searchable evidence…", expanded=True) as status:
                chunks, index = build_knowledge_base(uploaded_files, progress_callback=lambda value, text: progress.progress(value, text=text))
                st.session_state.knowledge_chunks = chunks
                st.session_state.index = index
                st.session_state.documents = [f.name for f in uploaded_files]
                st.session_state.kb_ready = True
                status.update(label="Knowledge base ready", state="complete", expanded=False)
            progress.empty()
            st.success(f"Indexed {len(chunks):,} evidence chunks from {len(uploaded_files)} document(s).", icon="✅")
        except Exception as error:
            progress.empty()
            st.error(f"Indexing failed: {error}. Check the PDF and Gemini connection, then try again.", icon="⚠️")

    st.markdown("<div class='section-label'>Ask the desk</div>", unsafe_allow_html=True)
    st.markdown("### What do you need to verify?")
    question = st.text_area("Your question", placeholder="e.g. How many casual leaves am I entitled to?", height=105, disabled=not ready, key="question-input")
    ask = st.button("Ask with evidence", type="primary", disabled=not ready or not question.strip(), key="ask-question-button", use_container_width=True)
    if ask:
        try:
            with st.spinner("Searching the index and drafting a cited answer…"):
                answer, sources = rag_query_safe(question.strip())
            st.session_state.history.append({"question": question.strip(), "answer": answer, "sources": sources, "time": datetime.now(timezone.utc).strftime("%H:%M UTC")})
        except Exception as error:
            st.error(f"The answer could not be generated: {error}", icon="⚠️")

    if not ready:
        st.markdown("<div class='panel'><div class='eyebrow'>Not ready yet</div><p style='margin:.45rem 0 0;color:#59656a'>Build your knowledge base first. Questions stay disabled until the evidence index is complete.</p></div>", unsafe_allow_html=True)
    elif not st.session_state.history:
        st.markdown("<div class='panel'><div class='eyebrow'>Try a focused question</div><p style='margin:.45rem 0 0;color:#59656a'>Ask about an entitlement, process, deadline, or policy exception.</p></div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-label'>Reading notes</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'><div class='eyebrow'>Grounding protocol</div><h3 style='margin:.25rem 0'>Evidence before confidence</h3><p style='color:#59656a;line-height:1.6'>Answers are generated from the three closest passages in your index. Open the source notes below each response to verify the wording yourself.</p><div class='mono' style='font-size:.72rem;color:#28624f'>● SOURCE-CITED OUTPUT</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'><div class='eyebrow'>Suggested prompts</div><p style='margin:.55rem 0'>What does the policy say about probation?</p><p style='margin:.55rem 0'>Which approvals are required?</p><p style='margin:.55rem 0'>What is the escalation process?</p></div>", unsafe_allow_html=True)


if st.session_state.history:
    latest = st.session_state.history[-1]
    citation_html = "".join(
        f"<span class='citation' data-testid='citation-button-{i}'>[{i + 1}] {escape(str(source['source']))} · p.{source['page']}</span>"
        for i, source in enumerate(latest["sources"])
    )
    st.markdown("<div class='section-label'>Latest finding</div>", unsafe_allow_html=True)
    safe_question = escape(str(latest["question"]))
    safe_answer = escape(str(latest["answer"])).replace("\n", "<br>")
    st.markdown(f"<div class='answer' data-testid='answer-panel'><div class='eyebrow'>Grounded answer · {escape(str(latest['time']))}</div><div class='question-quote'>“{safe_question}”</div><div style='font-size:1.05rem;line-height:1.7'>{safe_answer}</div><div style='margin-top:1.2rem'><span class='mono' style='font-size:.7rem;color:#59656a'>RETRIEVED EVIDENCE</span><br>{citation_html}</div></div>", unsafe_allow_html=True)
    citation_columns = st.columns(len(latest["sources"]))
    for i, (column, source) in enumerate(zip(citation_columns, latest["sources"])):
        with column:
            if st.button(f"Inspect source {i + 1}", key=f"citation-button-action-{i}", use_container_width=True):
                st.session_state.selected_source = i
    if st.session_state.selected_source is not None:
        selected_index = st.session_state.selected_source
        selected = latest["sources"][selected_index]
        selected_text = escape(str(selected["text"])).replace("\n", "<br>")
        st.markdown(f"<div class='panel' data-testid='selected-source-preview'><div class='eyebrow'>Selected evidence · source {selected_index + 1}</div><div class='mono' style='font-size:.72rem;color:#59656a'>{escape(str(selected['source']))} · PAGE {selected['page']}</div><div class='source-text' style='margin-top:.7rem'>{selected_text}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Inspect the record</div>", unsafe_allow_html=True)
    tabs = st.tabs([f"[{i + 1}] {source['source']} · p.{source['page']}" for i, source in enumerate(latest["sources"])])
    for tab, source in zip(tabs, latest["sources"]):
        with tab:
            safe_source = escape(str(source["text"])).replace("\n", "<br>")
            st.markdown(f"<div data-testid='source-passage-viewer'><div class='mono' style='font-size:.7rem;color:#59656a;margin-bottom:.6rem'>EXACT RETRIEVED PASSAGE · PAGE {source['page']}</div><div class='source-text'>{safe_source}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:3rem'></div><div class='mono' style='font-size:.68rem;color:#59656a;border-top:1px solid #d7d0c4;padding-top:1rem'>EVIDENCE DESK · RETRIEVAL-AUGMENTED GENERATION · DOCUMENTS REMAIN THE SOURCE OF TRUTH</div>", unsafe_allow_html=True)
