"""
Corporate Knowledge Base RAG — Web App
MBA462B — CIA-3 | Group 3

This is the website version of the Colab notebook. Same pipeline, same logic,
same two Gemini models — just wrapped in a Streamlit interface instead of
notebook cells, so anyone can open a link, upload PDFs, and ask questions.

Pipeline (identical to the notebook):
  Upload PDFs -> extract text -> chunk -> embed -> FAISS index   (offline, once)
  Ask question -> embed question -> retrieve nearest chunks -> Gemini answers  (online, per question)
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
st.set_page_config(page_title="Corporate Knowledge Base Assistant", page_icon="📚", layout="wide")
st.title("📚 Corporate Knowledge Base Assistant")
st.caption("Upload company policy PDFs, then ask questions in plain language. "
           "Answers are grounded strictly in your documents, with source citations — no hallucination.")

# ---------------------------------------------------------------------------
# Connect to Gemini using the API key stored in Streamlit Secrets
# (Settings -> Secrets on Streamlit Community Cloud; see deployment guide)
# ---------------------------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Settings -> Secrets, then reload.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash-lite"

# ---------------------------------------------------------------------------
# Session state — holds the knowledge base for this browser session
# ---------------------------------------------------------------------------
if "knowledge_chunks" not in st.session_state:
    st.session_state.knowledge_chunks = []      # list of {source, page, text}
if "index" not in st.session_state:
    st.session_state.index = None                # FAISS index
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

# ---------------------------------------------------------------------------
# Core pipeline functions — ported directly from the notebook
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file):
    """Reads a single uploaded PDF and returns (page_number, page_text) tuples."""
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text, chunk_size=400, overlap=50):
    """Splits text into overlapping word chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    """Calls Gemini's embedding model and returns a single embedding vector."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values


def build_knowledge_base(uploaded_files, progress_callback=None):
    """Full offline stage: extract -> chunk -> embed -> FAISS index."""
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
            time.sleep(1)  # stay comfortably within free-tier rate limits

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return knowledge_chunks, index


def retrieve_top_chunks(query, index, knowledge_chunks, top_k=3):
    """Embeds the question, then finds the top_k nearest chunks in the FAISS index."""
    query_vec = embed_text(query, task_type="RETRIEVAL_QUERY")
    query_vec = np.array([query_vec]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [knowledge_chunks[i] for i in indices[0]]


def generate_answer(query, retrieved_chunks):
    """Sends retrieved context + question to Gemini; answer is grounded strictly in that context."""
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
    """Full retrieve + generate pipeline, with automatic retry on rate-limit errors."""
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
# Sidebar — build / rebuild the knowledge base
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Build knowledge base")
    uploaded_files = st.file_uploader(
        "Upload policy PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Build / Rebuild knowledge base", type="primary", disabled=not uploaded_files):
        progress_bar = st.progress(0.0, text="Starting...")

        def _update(pct, msg):
            progress_bar.progress(pct, text=msg)

        with st.spinner("Reading and embedding documents..."):
            chunks, index = build_knowledge_base(uploaded_files, progress_callback=_update)

        st.session_state.knowledge_chunks = chunks
        st.session_state.index = index
        st.session_state.kb_ready = True
        progress_bar.empty()
        st.success(f"Knowledge base ready — {len(chunks)} chunks from {len(uploaded_files)} document(s).")

    if st.session_state.kb_ready:
        st.info(f"✅ {len(st.session_state.knowledge_chunks)} chunks loaded and searchable.")
    else:
        st.warning("Upload PDFs and build the knowledge base to start asking questions.")

# ---------------------------------------------------------------------------
# Main area — ask a question
# ---------------------------------------------------------------------------
st.header("2. Ask a question")

question = st.text_input(
    "Type your question about the uploaded policies",
    placeholder="e.g. How many casual leaves am I entitled to?",
    disabled=not st.session_state.kb_ready,
)

if st.button("Get answer", disabled=not st.session_state.kb_ready or not question.strip()):
    with st.spinner("Searching documents and generating answer..."):
        answer, sources = rag_query_safe(
            question, st.session_state.index, st.session_state.knowledge_chunks
        )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources used")
    for s in sources:
        with st.expander(f"📄 {s['source']} — Page {s['page']}"):
            st.write(s["text"])

st.divider()
st.caption(
    "Retrieval-Augmented Generation (RAG) prototype — MBA462B, CIA-3, Group 3. "
    "Answers are generated strictly from uploaded documents; no external knowledge is used."
)
