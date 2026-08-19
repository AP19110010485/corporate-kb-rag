import time
import html
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ClientError


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="KnowledgeHub AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PREMIUM UI / CSS
# =============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

:root {
    --primary: #6C5CE7;
    --primary-dark: #5144C7;
    --blue: #3B82F6;
    --cyan: #06B6D4;
    --green: #10B981;
    --orange: #F59E0B;
    --red: #EF4444;
    --text: #172033;
    --muted: #697386;
    --border: #E8EAF1;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(108,92,231,.08), transparent 24%),
        radial-gradient(circle at 10% 25%, rgba(59,130,246,.05), transparent 25%),
        #F7F8FC;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.25rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,.98);
    border-right: 1px solid #E9EAF1;
}

section[data-testid="stSidebar"] > div {
    padding: 1rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 6px 20px;
}

.sidebar-logo {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg,#6047D9,#9A7BFF);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 21px;
    box-shadow: 0 8px 20px rgba(108,92,231,.28);
}

.sidebar-brand-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    color: #182033;
    font-size: 16px;
    line-height: 1.1;
}

.sidebar-brand-sub {
    color: #8A91A4;
    font-size: 10px;
    margin-top: 4px;
}

/* Header */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
}

.greeting-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #182033;
}

.greeting-sub {
    color: #8991A5;
    font-size: .84rem;
    margin-top: 3px;
}

.notification {
    width: 45px;
    height: 45px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: white;
    border: 1px solid #ECEEF5;
    border-radius: 14px;
    box-shadow: 0 5px 20px rgba(0,0,0,.04);
    font-size: 21px;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    padding: 2.35rem 2.55rem;
    min-height: 270px;
    margin-bottom: 1.5rem;
    background:
        radial-gradient(circle at 85% 18%, rgba(255,255,255,.20), transparent 20%),
        linear-gradient(120deg,#5A3FD6 0%,#704DE8 38%,#4B74E8 100%);
    box-shadow: 0 25px 50px -25px rgba(75,65,180,.55);
}

.hero::before {
    content: "";
    position: absolute;
    width: 500px;
    height: 500px;
    right: -170px;
    top: -210px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 50%;
}

.hero::after {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    right: -90px;
    bottom: -245px;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 50%;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 760px;
}

.hero-eyebrow {
    display: inline-flex;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.17);
    color: white;
    font-size: .72rem;
    font-weight: 700;
    margin-bottom: 14px;
}

.hero h1 {
    color: white;
    font-family: 'Poppins', sans-serif;
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 12px;
}

.hero p {
    color: rgba(255,255,255,.9);
    font-size: .96rem;
    line-height: 1.65;
    max-width: 690px;
    margin: 0 0 19px;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.hero-badge {
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,.14);
    color: white;
    border: 1px solid rgba(255,255,255,.14);
    font-size: .70rem;
    font-weight: 600;
}

/* Section */
.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.45rem 0 .85rem;
}

.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.13rem;
    font-weight: 700;
    color: #182033;
}

.section-subtitle {
    color: #8A91A4;
    font-size: .76rem;
    margin-top: 3px;
}

/* Cards */
.stat-card,
.feature-card,
.document-card,
.workflow-card,
.recent-card,
.answer-wrapper,
.ask-panel,
.source-card,
.info-banner {
    background: rgba(255,255,255,.96);
    border: 1px solid #ECEEF5;
    box-shadow: 0 8px 28px rgba(31,35,50,.045);
}

.stat-card {
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    min-height: 120px;
    transition: .2s ease;
}

.stat-card:hover,
.feature-card:hover,
.document-card:hover,
.workflow-card:hover,
.recent-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 35px rgba(31,35,50,.09);
}

.stat-icon {
    width: 39px;
    height: 39px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.stat-value {
    font-family: 'Poppins', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #182033;
    margin-top: 9px;
}

.stat-label {
    font-size: .73rem;
    color: #858DA1;
}

.feature-card {
    border-radius: 18px;
    padding: 1.2rem;
    min-height: 165px;
    transition: .2s ease;
}

.feature-icon {
    width: 45px;
    height: 45px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 12px;
}

.feature-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #20283A;
    font-size: .88rem;
    margin-bottom: 6px;
}

.feature-text {
    color: #7B8497;
    font-size: .73rem;
    line-height: 1.55;
}

.ask-panel {
    border-radius: 20px;
    padding: 1.25rem;
    margin-bottom: .75rem;
}

.ask-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #20283A;
    font-size: .98rem;
}

.ask-subtitle {
    color: #8991A5;
    font-size: .74rem;
    margin-top: 3px;
}

.try-label {
    color: #8A91A4;
    font-size: .72rem;
    font-weight: 700;
    margin: 12px 0 7px;
}

.answer-wrapper {
    border: 1px solid #E7E3FF;
    border-radius: 22px;
    padding: 1.45rem;
    box-shadow: 0 12px 35px rgba(108,92,231,.07);
}

.answer-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}

.ai-avatar {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg,#6C5CE7,#9A7BFF);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
}

.answer-label {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #242A3D;
    font-size: .9rem;
}

.answer-status {
    font-size: .67rem;
    color: #10A56A;
    margin-top: 2px;
}

.answer-question {
    color: #6C5CE7;
    font-size: .75rem;
    font-weight: 600;
    margin-bottom: .7rem;
}

.answer-text {
    color: #333A4D;
    font-size: .93rem;
    line-height: 1.72;
}

.source-card {
    background: #F8F7FF;
    border: 1px solid #E9E5FF;
    border-radius: 14px;
    padding: .85rem;
    margin-bottom: .55rem;
}

.source-name {
    color: #5D4ED0;
    font-weight: 700;
    font-size: .76rem;
}

.source-meta {
    color: #8A91A4;
    font-size: .67rem;
    margin-top: 3px;
}

.recent-card {
    border-radius: 17px;
    padding: .9rem 1rem;
    transition: .2s ease;
}

.recent-question {
    color: #293146;
    font-size: .77rem;
    font-weight: 600;
}

.recent-meta {
    color: #9299AA;
    font-size: .66rem;
    margin-top: 4px;
}

.document-card {
    border-radius: 18px;
    padding: 1.1rem;
    transition: .2s ease;
}

.document-icon {
    width: 44px;
    height: 44px;
    background: #FFF1F2;
    color: #EF4444;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 10px;
}

.document-name {
    color: #252C40;
    font-weight: 700;
    font-size: .81rem;
    word-break: break-word;
}

.document-info {
    color: #8A91A4;
    font-size: .68rem;
    margin-top: 4px;
}

.workflow-card {
    border-radius: 20px;
    padding: 1.3rem;
    min-height: 190px;
    transition: .2s ease;
}

.workflow-number {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    background: linear-gradient(135deg,#6C5CE7,#8E7CFF);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    margin-bottom: 12px;
}

.workflow-title {
    font-family: 'Poppins', sans-serif;
    color: #242B3D;
    font-weight: 700;
    font-size: .88rem;
}

.workflow-text {
    color: #7D8699;
    font-size: .73rem;
    line-height: 1.55;
    margin-top: 7px;
}

.info-banner {
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    background: linear-gradient(100deg,#fff,#F7F5FF);
}

.info-banner-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #272E40;
    font-size: .88rem;
}

.info-banner-text {
    color: #788194;
    font-size: .74rem;
    line-height: 1.6;
    margin-top: 5px;
}

.future-card {
    background: linear-gradient(135deg,#171D4A,#312A72);
    color: white;
    border-radius: 22px;
    padding: 1.4rem;
    min-height: 145px;
    box-shadow: 0 20px 45px rgba(31,29,94,.2);
}

.future-icon {
    font-size: 22px;
    margin-bottom: 9px;
}

.future-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: .86rem;
}

.future-text {
    color: rgba(255,255,255,.72);
    font-size: .70rem;
    line-height: 1.5;
    margin-top: 5px;
}

.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    background: white;
    border: 1px dashed #DCDDE7;
    border-radius: 20px;
}

.empty-icon {
    font-size: 42px;
    margin-bottom: 9px;
}

.empty-title {
    font-family: 'Poppins', sans-serif;
    color: #30374A;
    font-weight: 700;
}

.empty-text {
    color: #8A91A4;
    font-size: .77rem;
}

/* Inputs and buttons */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 13px !important;
    border: 1px solid #E2E4EC !important;
    background: white !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    border-color: #8B7CF6 !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,.10) !important;
}

div.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    border: 1px solid #E2E4EC !important;
    transition: all .2s ease !important;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(108,92,231,.14);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg,#6547E7,#8D6CF4) !important;
    border: none !important;
    color: white !important;
}

[data-testid="stFileUploader"] {
    background: #FAFAFE;
    border-radius: 14px;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #D7D3F5 !important;
    border-radius: 14px !important;
}

[data-testid="stExpander"] {
    border: 1px solid #E9EAF2 !important;
    border-radius: 14px !important;
    background: white !important;
}

.kb-ready {
    background: #EAF8F0;
    color: #18834B;
    border: 1px solid #CBEFD9;
    border-radius: 14px;
    padding: 10px 12px;
    font-size: .78rem;
    font-weight: 700;
}

.kb-empty {
    background: #FFF7E6;
    color: #A86F08;
    border: 1px solid #FFE6B3;
    border-radius: 14px;
    padding: 10px 12px;
    font-size: .78rem;
    font-weight: 700;
}

@media (max-width: 800px) {
    .hero h1 {
        font-size: 1.8rem;
    }
    .hero {
        padding: 1.7rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# GEMINI CONFIGURATION
# =============================================================================

if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "⚠️ Gemini API key not found. Add GEMINI_API_KEY "
        "under Streamlit → Settings → Secrets."
    )
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"


# =============================================================================
# SESSION STATE
# =============================================================================

defaults = {
    "knowledge_chunks": [],
    "index": None,
    "kb_ready": False,
    "history": [],
    "uploaded_documents": [],
    "active_page": "Home",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# RAG FUNCTIONS
# =============================================================================

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text))

    return pages


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0

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
        config=types.EmbedContentConfig(
            task_type=task_type
        ),
    )
    return result.embeddings[0].values


def build_knowledge_base(uploaded_files, progress_callback=None):
    knowledge_chunks = []

    for file in uploaded_files:
        pages = extract_text_from_pdf(file)

        for page_number, page_text in pages:
            if not page_text.strip():
                continue

            for chunk in chunk_text(page_text):
                knowledge_chunks.append(
                    {
                        "source": file.name,
                        "page": page_number,
                        "text": chunk,
                    }
                )

    if not knowledge_chunks:
        raise ValueError(
            "No readable text was found in the uploaded PDFs. "
            "If the PDF is scanned/image-based, OCR support is required."
        )

    embeddings = []
    total = len(knowledge_chunks)

    for i, chunk in enumerate(knowledge_chunks):
        embeddings.append(embed_text(chunk["text"]))

        if progress_callback:
            progress_callback(
                (i + 1) / total,
                f"Embedding chunk {i + 1}/{total}..."
            )

        if i % 20 == 0 and i > 0:
            time.sleep(1)

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return knowledge_chunks, index


def retrieve_top_chunks(query, index, knowledge_chunks, top_k=4):
    query_vector = embed_text(
        query,
        task_type="RETRIEVAL_QUERY"
    )

    query_vector = np.asarray(
        [query_vector],
        dtype="float32"
    )

    top_k = min(top_k, len(knowledge_chunks))

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for index_number in indices[0]:
        if index_number >= 0:
            results.append(
                knowledge_chunks[index_number]
            )

    return results


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {chunk['source']} | Page {chunk['page']}]\n"
        f"{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a corporate knowledge base AI assistant.

Answer the user's question ONLY from the retrieved context.

Rules:
- Do not use outside knowledge.
- Do not guess or invent information.
- If the answer is not present, say:
  "I couldn't find this information in the uploaded knowledge base."
- Keep the answer concise and easy to understand.
- Mention the relevant source filename and page number.
- Never invent a policy, number, date, or procedure.

Retrieved context:
{context}

User question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )

    return response.text


def rag_query_safe(query, index, knowledge_chunks, top_k=4, max_retries=3):
    for attempt in range(max_retries):
        try:
            retrieved = retrieve_top_chunks(
                query,
                index,
                knowledge_chunks,
                top_k
            )

            answer = generate_answer(
                query,
                retrieved
            )

            return answer, retrieved

        except ClientError as error:
            if "RESOURCE_EXHAUSTED" in str(error):
                if attempt == max_retries - 1:
                    raise

                st.warning(
                    f"Gemini rate limit reached. Retrying "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(20)
            else:
                raise

    raise RuntimeError("Unable to complete the RAG request.")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">
    <div class="sidebar-logo">✨</div>
    <div>
        <div class="sidebar-brand-title">KnowledgeHub</div>
        <div class="sidebar-brand-sub">AI Knowledge Assistant</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")

    navigation = [
        ("🏠", "Home"),
        ("🤖", "Ask AI"),
        ("📚", "Knowledge Base"),
        ("🔎", "Sources"),
        ("📊", "Analytics"),
        ("🧠", "How RAG Works"),
    ]

    for icon, label in navigation:
        if st.button(
            f"{icon}  {label}",
            key=f"navigation_{label}",
            use_container_width=True,
        ):
            st.session_state.active_page = label
            st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload company PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload policies, handbooks, SOPs and other company documents.",
    )

    if st.button(
        "🚀  Build Knowledge Base",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        progress = st.progress(
            0.0,
            text="Preparing documents..."
        )

        def update_progress(value, message):
            progress.progress(value, text=message)

        try:
            with st.spinner("Building your AI knowledge base..."):
                chunks, index = build_knowledge_base(
                    uploaded_files,
                    progress_callback=update_progress
                )

            st.session_state.knowledge_chunks = chunks
            st.session_state.index = index
            st.session_state.kb_ready = True
            st.session_state.uploaded_documents = [
                file.name for file in uploaded_files
            ]

            progress.empty()
            st.success("Knowledge base built successfully!")
            st.balloons()

        except Exception as error:
            progress.empty()
            st.error(f"Unable to build knowledge base: {error}")

    if st.session_state.kb_ready:
        st.markdown(
            f"""
<div class="kb-ready">
    🟢 Knowledge Base Ready<br>
    <span style="font-weight:500;font-size:.68rem;">
        {len(st.session_state.knowledge_chunks)} searchable chunks
    </span>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="kb-empty">
    🟡 Knowledge Base not built<br>
    <span style="font-weight:500;font-size:.68rem;">
        Upload PDFs and build your knowledge base
    </span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(
        """
<div style="
    padding:12px;
    border-radius:14px;
    background:linear-gradient(135deg,#F4F0FF,#EEF4FF);
    border:1px solid #E4E1FA;
">
    <div style="font-weight:700;font-size:12px;color:#343A55;">
        🎓 MBA462B · CIA-3
    </div>
    <div style="font-size:10px;color:#858CA0;margin-top:3px;">
        Corporate Knowledge Base RAG · Group 3
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# COMMON HEADER
# =============================================================================

st.html(
    """
<div class="top-header">
    <div>
        <div class="greeting-title">Good evening 👋</div>
        <div class="greeting-sub">
            Your AI-powered organizational knowledge assistant is ready.
        </div>
    </div>
    <div class="notification">🔔</div>
</div>
"""
)


# =============================================================================
# HERO
# =============================================================================

st.html(
    """
<div class="hero">
    <div class="hero-content">
        <div class="hero-eyebrow">✨ AI-POWERED KNOWLEDGE PLATFORM</div>

        <h1>Corporate Knowledge<br>Base Assistant</h1>

        <p>
            Ask questions in natural language and get intelligent,
            document-grounded answers from your organization's trusted
            knowledge base.
        </p>

        <div class="badge-row">
            <span class="hero-badge">🔍 Retrieval-Augmented Generation</span>
            <span class="hero-badge">📎 Source-Cited Answers</span>
            <span class="hero-badge">⚡ Powered by Gemini</span>
            <span class="hero-badge">🔐 Document Grounded</span>
        </div>
    </div>
</div>
"""
)


# =============================================================================
# HOME
# =============================================================================

def render_home():

    document_count = len(st.session_state.uploaded_documents)
    chunk_count = len(st.session_state.knowledge_chunks)
    question_count = len(st.session_state.history)
    source_count = len(
        set(
            chunk["source"]
            for chunk in st.session_state.knowledge_chunks
        )
    )

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">Knowledge Overview</div>
        <div class="section-subtitle">
            Your organization's AI knowledge ecosystem
        </div>
    </div>
</div>
"""
    )

    stats = [
        ("📄", document_count, "Documents", "#F0EDFF", "#6C5CE7"),
        ("🧩", chunk_count, "Knowledge Chunks", "#EAF8F4", "#10B981"),
        ("💬", question_count, "Questions Asked", "#FFF5E8", "#F59E0B"),
        ("🔗", source_count, "Active Sources", "#EDF5FF", "#3B82F6"),
    ]

    columns = st.columns(4)

    for column, (icon, value, label, background, color) in zip(
        columns,
        stats
    ):
        with column:
            st.html(
                f"""
<div class="stat-card">
    <div class="stat-icon" style="background:{background};color:{color};">
        {icon}
    </div>
    <div class="stat-value">{value}</div>
    <div class="stat-label">{label}</div>
</div>
"""
            )

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">Ask your knowledge base</div>
        <div class="section-subtitle">
            Search trusted company documents using natural language
        </div>
    </div>
</div>

<div class="ask-panel">
    <div class="ask-title">🤖 What would you like to know?</div>
    <div class="ask-subtitle">
        The assistant retrieves relevant information before generating an answer.
    </div>
</div>
"""
    )

    q_col, button_col = st.columns([5, 1])

    with q_col:
        home_question = st.text_input(
            "Home question",
            placeholder="e.g. What is the company's maternity leave policy?",
            label_visibility="collapsed",
            disabled=not st.session_state.kb_ready,
        )

    with button_col:
        home_ask = st.button(
            "Ask AI ✨",
            type="primary",
            use_container_width=True,
            disabled=(
                not st.session_state.kb_ready
                or not home_question.strip()
            ),
        )

    if home_ask:
        with st.spinner("Searching your knowledge base..."):
            answer, sources = rag_query_safe(
                home_question,
                st.session_state.index,
                st.session_state.knowledge_chunks,
            )

        st.session_state.history.append(
            (home_question, answer, sources)
        )
        st.session_state.active_page = "Ask AI"
        st.rerun()

    st.html('<div class="try-label">Try asking</div>')

    suggestions = [
        "What is the company's maternity leave policy?",
        "How many casual leaves are employees entitled to?",
        "What is the employee onboarding process?",
        "How can I request IT access?",
    ]

    suggestion_columns = st.columns(4)

    for column, suggestion in zip(
        suggestion_columns,
        suggestions
    ):
        with column:
            if st.button(
                suggestion,
                key=f"suggestion_{suggestion}",
                use_container_width=True,
                disabled=not st.session_state.kb_ready,
            ):
                with st.spinner("Searching documents..."):
                    answer, sources = rag_query_safe(
                        suggestion,
                        st.session_state.index,
                        st.session_state.knowledge_chunks,
                    )

                st.session_state.history.append(
                    (suggestion, answer, sources)
                )
                st.session_state.active_page = "Ask AI"
                st.rerun()

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">Intelligent by design</div>
        <div class="section-subtitle">
            Built around the core principles of your RAG project
        </div>
    </div>
</div>
"""
    )

    features = [
        (
            "🧠",
            "Advanced RAG",
            "Retrieval-Augmented Generation finds relevant information before the AI generates a response.",
            "#F0EDFF",
            "#6C5CE7",
        ),
        (
            "📚",
            "Smart Sources",
            "Answers are grounded in uploaded documents with transparent source references.",
            "#EAF8F4",
            "#10B981",
        ),
        (
            "🔐",
            "Document Grounded",
            "The assistant is instructed not to guess when information cannot be found.",
            "#FFF5E8",
            "#F59E0B",
        ),
        (
            "⚡",
            "Gemini Powered",
            "Google Gemini handles embeddings and natural-language answer generation.",
            "#EDF5FF",
            "#3B82F6",
        ),
    ]

    feature_columns = st.columns(4)

    for column, (icon, title, text, background, color) in zip(
        feature_columns,
        features
    ):
        with column:
            st.html(
                f"""
<div class="feature-card">
    <div class="feature-icon"
         style="background:{background};color:{color};">
        {icon}
    </div>
    <div class="feature-title">{title}</div>
    <div class="feature-text">{text}</div>
</div>
"""
            )

    if st.session_state.history:
        st.html(
            """
<div class="section-header">
    <div>
        <div class="section-title">Recent Questions</div>
        <div class="section-subtitle">
            Your latest knowledge-base interactions
        </div>
    </div>
</div>
"""
        )

        for question, answer, sources in reversed(
            st.session_state.history[-5:]
        ):
            safe_question = html.escape(question)

            st.html(
                f"""
<div class="recent-card">
    <div class="recent-question">💬 {safe_question}</div>
    <div class="recent-meta">
        {len(sources)} source(s) retrieved
    </div>
</div>
"""
            )


# =============================================================================
# ASK AI
# =============================================================================

def render_ask_ai():

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">🤖 Ask AI</div>
        <div class="section-subtitle">
            Get answers grounded in your organization's documents
        </div>
    </div>
</div>
"""
    )

    if not st.session_state.kb_ready:
        st.html(
            """
<div class="empty-state">
    <div class="empty-icon">📚</div>
    <div class="empty-title">Your knowledge base is empty</div>
    <div class="empty-text">
        Upload company PDFs from the sidebar and build your knowledge base
        to start asking questions.
    </div>
</div>
"""
        )
        return

    question = st.text_area(
        "Ask your question",
        placeholder="Ask anything about your uploaded company documents...",
        height=110,
    )

    ask = st.button(
        "✨ Generate Answer",
        type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    )

    if ask:
        with st.spinner(
            "Retrieving relevant knowledge and generating answer..."
        ):
            answer, sources = rag_query_safe(
                question,
                st.session_state.index,
                st.session_state.knowledge_chunks,
            )

        st.session_state.history.append(
            (question, answer, sources)
        )

    if st.session_state.history:
        latest_question, latest_answer, latest_sources = (
            st.session_state.history[-1]
        )

        safe_question = html.escape(latest_question)
        safe_answer = html.escape(latest_answer).replace("\n", "<br>")

        st.html(
            f"""
<div style="height:16px;"></div>

<div class="answer-wrapper">
    <div class="answer-header">
        <div class="ai-avatar">✨</div>
        <div>
            <div class="answer-label">KnowledgeHub AI</div>
            <div class="answer-status">● Grounded response</div>
        </div>
    </div>

    <div class="answer-question">Q · {safe_question}</div>

    <div class="answer-text">
        {safe_answer}
    </div>
</div>
"""
        )

        st.html(
            """
<div class="section-header">
    <div>
        <div class="section-title">📎 Sources used</div>
        <div class="section-subtitle">
            Retrieved document passages supporting the answer
        </div>
    </div>
</div>
"""
        )

        source_columns = st.columns(
            min(len(latest_sources), 4)
        )

        for column, source in zip(
            source_columns,
            latest_sources
        ):
            with column:
                safe_source = html.escape(source["source"])

                st.html(
                    f"""
<div class="source-card">
    <div class="source-name">📄 {safe_source}</div>
    <div class="source-meta">Page {source["page"]}</div>
</div>
"""
                )

        with st.expander("🔎 View retrieved source passages"):
            for source in latest_sources:
                st.markdown(
                    f"**{source['source']} — Page {source['page']}**"
                )
                st.write(source["text"])
                st.divider()


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

def render_knowledge_base():

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">📚 Knowledge Base</div>
        <div class="section-subtitle">
            Manage and explore the documents powering your assistant
        </div>
    </div>
</div>
"""
    )

    if not st.session_state.kb_ready:
        st.html(
            """
<div class="empty-state">
    <div class="empty-icon">📄</div>
    <div class="empty-title">No documents indexed yet</div>
    <div class="empty-text">
        Upload PDFs using the sidebar to create your knowledge base.
    </div>
</div>
"""
        )
        return

    documents = st.session_state.uploaded_documents

    document_columns = st.columns(3)

    for column, document in zip(
        document_columns,
        documents
    ):
        with column:
            document_chunks = [
                chunk
                for chunk in st.session_state.knowledge_chunks
                if chunk["source"] == document
            ]

            pages = len(
                set(
                    chunk["page"]
                    for chunk in document_chunks
                )
            )

            safe_document = html.escape(document)

            st.html(
                f"""
<div class="document-card">
    <div class="document-icon">📄</div>
    <div class="document-name">{safe_document}</div>
    <div class="document-info">
        {pages} pages · {len(document_chunks)} chunks
    </div>
    <div style="
        margin-top:12px;
        color:#10A56A;
        font-size:10px;
        font-weight:700;
    ">
        ● INDEXED & SEARCHABLE
    </div>
</div>

<div style="height:10px;"></div>
"""
            )

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">Knowledge Categories</div>
        <div class="section-subtitle">
            Suggested organizational structure for the project
        </div>
    </div>
</div>
"""
    )

    categories = [
        ("👥", "HR & Operations", "Policies, onboarding, benefits and employee support"),
        ("💻", "IT Knowledge", "Troubleshooting, access requests and IT policies"),
        ("💰", "Finance", "Expenses, reimbursement, budget and compliance"),
        ("🤝", "Sales", "Product pricing, proposals and competitor intelligence"),
        ("📖", "Handbooks", "Employee handbooks and organizational guidance"),
        ("⚙️", "SOPs", "Standard operating procedures and process documentation"),
    ]

    category_columns = st.columns(3)

    for index, category in enumerate(categories):
        column = category_columns[index % 3]

        with column:
            icon, title, description = category

            st.html(
                f"""
<div class="feature-card">
    <div class="feature-icon"
         style="background:#F1F0FF;color:#6C5CE7;">
        {icon}
    </div>
    <div class="feature-title">{title}</div>
    <div class="feature-text">{description}</div>
</div>

<div style="height:10px;"></div>
"""
            )


# =============================================================================
# SOURCES
# =============================================================================

def render_sources():

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">🔎 Source Management</div>
        <div class="section-subtitle">
            See where your AI knowledge comes from
        </div>
    </div>
</div>
"""
    )

    if not st.session_state.kb_ready:
        st.info("Upload and index documents to view sources.")
        return

    sources = {}

    for chunk in st.session_state.knowledge_chunks:
        source_name = chunk["source"]
        sources.setdefault(source_name, []).append(chunk)

    for source_name, chunks in sources.items():
        pages = sorted(
            set(chunk["page"] for chunk in chunks)
        )

        with st.expander(
            f"📄 {source_name} · {len(pages)} pages"
        ):
            st.write(
                f"**Indexed chunks:** {len(chunks)}"
            )
            st.write(
                f"**Pages represented:** "
                f"{', '.join(map(str, pages))}"
            )

            for chunk in chunks[:3]:
                st.markdown(
                    f"**Page {chunk['page']}**"
                )

                preview = chunk["text"][:500]

                if len(chunk["text"]) > 500:
                    preview += "..."

                st.caption(preview)


# =============================================================================
# ANALYTICS
# =============================================================================

def render_analytics():

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">📊 Knowledge Analytics</div>
        <div class="section-subtitle">
            Understand how your knowledge assistant is being used
        </div>
    </div>
</div>
"""
    )

    analytics = [
        (
            len(st.session_state.history),
            "Total Questions"
        ),
        (
            len(st.session_state.knowledge_chunks),
            "Indexed Chunks"
        ),
        (
            len(st.session_state.uploaded_documents),
            "Documents"
        ),
        (
            len(
                set(
                    chunk["source"]
                    for chunk in st.session_state.knowledge_chunks
                )
            ),
            "Active Sources"
        ),
    ]

    columns = st.columns(4)

    for column, (value, label) in zip(
        columns,
        analytics
    ):
        with column:
            st.html(
                f"""
<div class="stat-card">
    <div class="stat-value">{value}</div>
    <div class="stat-label">{label}</div>
</div>
"""
            )

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">Retrieval Activity</div>
        <div class="section-subtitle">
            Questions and source retrieval activity during this session
        </div>
    </div>
</div>
"""
    )

    if not st.session_state.history:
        st.info("Ask some questions to populate analytics.")
        return

    for number, (question, answer, sources) in enumerate(
        reversed(st.session_state.history),
        start=1
    ):
        safe_question = html.escape(question)

        st.html(
            f"""
<div class="recent-card">
    <div class="recent-question">
        {number}. {safe_question}
    </div>
    <div class="recent-meta">
        {len(sources)} source(s) retrieved
    </div>
</div>

<div style="height:7px;"></div>
"""
        )


# =============================================================================
# HOW RAG WORKS
# =============================================================================

def render_rag():

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">🧠 How RAG Works</div>
        <div class="section-subtitle">
            From employee question to grounded AI answer
        </div>
    </div>
</div>

<div class="info-banner">
    <div class="info-banner-title">
        Retrieval-Augmented Generation
    </div>
    <div class="info-banner-text">
        Instead of relying only on the language model's pre-trained knowledge,
        the system retrieves relevant information from organizational
        documents and uses that context to generate the answer.
    </div>
</div>
"""
    )

    workflows = [
        ("1", "Employee asks", "The user asks a question in natural language."),
        ("2", "Semantic search", "The question is converted into an embedding and compared with document embeddings."),
        ("3", "Retrieve chunks", "FAISS returns the most relevant knowledge chunks."),
        ("4", "Send context to LLM", "Relevant document content is provided to Gemini."),
        ("5", "Generate answer", "Gemini creates a concise natural-language response."),
        ("6", "Return sources", "The answer is shown with source documents and page references."),
    ]

    columns = st.columns(3)

    for index, workflow in enumerate(workflows):
        column = columns[index % 3]

        with column:
            number, title, description = workflow

            st.html(
                f"""
<div class="workflow-card">
    <div class="workflow-number">{number}</div>
    <div class="workflow-title">{title}</div>
    <div class="workflow-text">{description}</div>
</div>

<div style="height:10px;"></div>
"""
            )

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">Technology Stack</div>
        <div class="section-subtitle">
            Technologies used in the project architecture
        </div>
    </div>
</div>
"""
    )

    technologies = [
        ("💬", "NLP"),
        ("🔎", "Semantic Search"),
        ("🧩", "FAISS"),
        ("🧠", "RAG"),
        ("⚡", "Gemini"),
        ("📄", "PDF Knowledge"),
    ]

    tech_columns = st.columns(6)

    for column, (icon, name) in zip(
        tech_columns,
        technologies
    ):
        with column:
            st.html(
                f"""
<div class="feature-card" style="min-height:110px;text-align:center;">
    <div style="font-size:24px;">{icon}</div>
    <div class="feature-title" style="margin-top:8px;">
        {name}
    </div>
</div>
"""
            )

    st.html(
        """
<div class="section-header">
    <div>
        <div class="section-title">🚀 Future Scope</div>
        <div class="section-subtitle">
            Expansion opportunities for the assistant
        </div>
    </div>
</div>
"""
    )

    future = [
        ("🎙️", "Voice AI", "Ask questions and receive answers using voice."),
        ("🌍", "Multi-language", "Support multiple languages for diverse users."),
        ("💬", "Teams & Slack", "Bring knowledge assistance into collaboration platforms."),
        ("👤", "Personalized Accounts", "Personalized experiences and query history."),
        ("📱", "Mobile Application", "Access organizational knowledge from anywhere."),
        ("🔄", "Real-time Sync", "Keep answers aligned with updated documents."),
    ]

    future_columns = st.columns(3)

    for index, item in enumerate(future):
        column = future_columns[index % 3]

        with column:
            icon, title, description = item

            st.html(
                f"""
<div class="future-card">
    <div class="future-icon">{icon}</div>
    <div class="future-title">{title}</div>
    <div class="future-text">{description}</div>
</div>

<div style="height:10px;"></div>
"""
            )


# =============================================================================
# PAGE ROUTING
# =============================================================================

page = st.session_state.active_page

if page == "Home":
    render_home()

elif page == "Ask AI":
    render_ask_ai()

elif page == "Knowledge Base":
    render_knowledge_base()

elif page == "Sources":
    render_sources()

elif page == "Analytics":
    render_analytics()

elif page == "How RAG Works":
    render_rag()


# =============================================================================
# FOOTER
# =============================================================================

st.html(
    """
<div style="
    margin-top:50px;
    padding-top:18px;
    border-top:1px solid #E8E9F0;
    text-align:center;
    color:#9AA1B2;
    font-size:11px;
">
    ✨ KnowledgeHub AI · Corporate Knowledge Base RAG
    · MBA462B CIA-3 · Group 3
    <br>
    Answers are generated strictly from indexed documents.
</div>
"""
)
