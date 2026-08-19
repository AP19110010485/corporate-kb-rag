import time
import html
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ClientError


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="KnowledgeHub AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# GLOBAL CSS — PREMIUM SAAS UI
# ============================================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

:root {
    --primary: #6C5CE7;
    --primary-dark: #5144c7;
    --secondary: #8B5CF6;
    --blue: #3B82F6;
    --cyan: #06B6D4;
    --green: #10B981;
    --orange: #F59E0B;
    --red: #EF4444;
    --text: #172033;
    --muted: #697386;
    --border: #E8EAF1;
    --surface: #FFFFFF;
    --background: #F6F7FB;
}

/* -------------------------------------------------------------------------
   GLOBAL
------------------------------------------------------------------------- */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 5%, rgba(108,92,231,0.08), transparent 25%),
        radial-gradient(circle at 15% 20%, rgba(59,130,246,0.05), transparent 25%),
        #F7F8FC;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* -------------------------------------------------------------------------
   SIDEBAR
------------------------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.96);
    border-right: 1px solid #E9EAF1;
}

section[data-testid="stSidebar"] > div {
    padding: 1.2rem 1rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 8px 24px 8px;
}

.sidebar-logo {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #6C5CE7, #9B7CFF);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    box-shadow: 0 8px 20px rgba(108,92,231,0.30);
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
    font-size: 11px;
    margin-top: 3px;
}


/* -------------------------------------------------------------------------
   TOP HEADER
------------------------------------------------------------------------- */

.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
    animation: fadeIn 0.45s ease;
}

.greeting-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #182033;
    margin-bottom: 3px;
}

.greeting-sub {
    color: #8991A5;
    font-size: 0.85rem;
}


/* -------------------------------------------------------------------------
   HERO
------------------------------------------------------------------------- */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    min-height: 285px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.5rem;

    background:
        radial-gradient(circle at 85% 20%, rgba(255,255,255,0.20), transparent 20%),
        radial-gradient(circle at 15% 80%, rgba(255,255,255,0.10), transparent 25%),
        linear-gradient(120deg, #5A3FD6 0%, #704DE8 38%, #4B74E8 100%);

    box-shadow:
        0 25px 50px -25px rgba(75,65,180,0.55);

    animation: fadeIn 0.6s ease;
}

.hero:before {
    content: "";
    position: absolute;
    width: 500px;
    height: 500px;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 50%;
    right: -150px;
    top: -180px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 380px;
    height: 380px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 50%;
    right: -80px;
    bottom: -240px;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 760px;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.17);
    color: rgba(255,255,255,0.95);
    font-size: 0.74rem;
    font-weight: 600;
    margin-bottom: 15px;
}

.hero h1 {
    font-family: 'Poppins', sans-serif;
    color: white;
    font-size: 2.45rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 12px 0;
}

.hero p {
    color: rgba(255,255,255,0.90);
    font-size: 0.98rem;
    line-height: 1.65;
    max-width: 690px;
    margin-bottom: 20px;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.hero-badge {
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    color: white;
    border: 1px solid rgba(255,255,255,0.14);
    font-size: 0.72rem;
    font-weight: 600;
    backdrop-filter: blur(8px);
}


/* -------------------------------------------------------------------------
   SECTION HEADERS
------------------------------------------------------------------------- */

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.6rem 0 0.9rem;
}

.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: #182033;
}

.section-subtitle {
    color: #8A91A4;
    font-size: 0.78rem;
}


/* -------------------------------------------------------------------------
   STAT CARDS
------------------------------------------------------------------------- */

.stat-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #ECEEF5;
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    min-height: 125px;
    box-shadow: 0 8px 30px rgba(31,35,50,0.045);
    transition: all 0.22s ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 35px rgba(31,35,50,0.09);
}

.stat-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
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
    margin-top: 10px;
}

.stat-label {
    font-size: 0.75rem;
    color: #858DA1;
}


/* -------------------------------------------------------------------------
   FEATURE CARDS
------------------------------------------------------------------------- */

.feature-card {
    background: white;
    border: 1px solid #ECEEF5;
    border-radius: 18px;
    padding: 1.25rem;
    min-height: 175px;
    transition: all 0.22s ease;
    box-shadow: 0 8px 25px rgba(31,35,50,0.04);
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 40px rgba(31,35,50,0.09);
}

.feature-icon {
    width: 46px;
    height: 46px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    margin-bottom: 13px;
}

.feature-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #20283A;
    font-size: 0.92rem;
    margin-bottom: 7px;
}

.feature-text {
    color: #7B8497;
    font-size: 0.76rem;
    line-height: 1.55;
}


/* -------------------------------------------------------------------------
   ASK AI PANEL
------------------------------------------------------------------------- */

.ask-panel {
    background: white;
    border: 1px solid #E9EAF2;
    border-radius: 22px;
    padding: 1.35rem;
    box-shadow: 0 12px 35px rgba(31,35,50,0.055);
}

.ask-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #20283A;
}

.ask-subtitle {
    color: #8991A5;
    font-size: 0.75rem;
    margin-top: 3px;
}

.try-label {
    color: #8A91A4;
    font-size: 0.73rem;
    font-weight: 600;
    margin-top: 12px;
}


/* -------------------------------------------------------------------------
   ANSWER
------------------------------------------------------------------------- */

.answer-wrapper {
    background: white;
    border: 1px solid #E7E3FF;
    border-radius: 22px;
    padding: 1.5rem;
    box-shadow: 0 12px 35px rgba(108,92,231,0.07);
    animation: slideUp 0.4s ease;
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
    background: linear-gradient(135deg, #6C5CE7, #9A7BFF);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
}

.answer-label {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #242A3D;
    font-size: 0.9rem;
}

.answer-status {
    font-size: 0.68rem;
    color: #10A56A;
    margin-top: 2px;
}

.answer-question {
    color: #6C5CE7;
    font-size: 0.76rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.answer-text {
    color: #333A4D;
    font-size: 0.94rem;
    line-height: 1.72;
}


/* -------------------------------------------------------------------------
   SOURCE CARDS
------------------------------------------------------------------------- */

.source-card {
    background: #F8F7FF;
    border: 1px solid #E9E5FF;
    border-radius: 14px;
    padding: 0.85rem;
    margin-bottom: 0.55rem;
}

.source-name {
    color: #5D4ED0;
    font-weight: 700;
    font-size: 0.77rem;
}

.source-meta {
    color: #8A91A4;
    font-size: 0.68rem;
    margin-top: 3px;
}


/* -------------------------------------------------------------------------
   RECENT QUESTIONS
------------------------------------------------------------------------- */

.recent-card {
    background: white;
    border: 1px solid #ECEEF5;
    border-radius: 17px;
    padding: 0.9rem 1rem;
    transition: all 0.2s ease;
}

.recent-card:hover {
    border-color: #DCD5FF;
    transform: translateX(2px);
}

.recent-question {
    color: #293146;
    font-size: 0.78rem;
    font-weight: 600;
}

.recent-meta {
    color: #9299AA;
    font-size: 0.67rem;
    margin-top: 4px;
}


/* -------------------------------------------------------------------------
   DOCUMENT CARDS
------------------------------------------------------------------------- */

.document-card {
    background: white;
    border: 1px solid #ECEEF5;
    border-radius: 18px;
    padding: 1.1rem;
    transition: 0.2s ease;
}

.document-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(31,35,50,0.07);
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
    font-size: 0.82rem;
    word-break: break-word;
}

.document-info {
    color: #8A91A4;
    font-size: 0.69rem;
    margin-top: 4px;
}


/* -------------------------------------------------------------------------
   WORKFLOW
------------------------------------------------------------------------- */

.workflow-card {
    background: white;
    border: 1px solid #E9EAF2;
    border-radius: 20px;
    padding: 1.3rem;
    min-height: 210px;
    box-shadow: 0 8px 25px rgba(31,35,50,0.04);
}

.workflow-number {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6C5CE7, #8E7CFF);
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
    font-size: 0.9rem;
}

.workflow-text {
    color: #7D8699;
    font-size: 0.74rem;
    line-height: 1.55;
    margin-top: 7px;
}


/* -------------------------------------------------------------------------
   INFO BANNERS
------------------------------------------------------------------------- */

.info-banner {
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    margin: 1rem 0;
    border: 1px solid #E9EAF2;
    background: linear-gradient(100deg, #FFFFFF, #F7F5FF);
}

.info-banner-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #272E40;
    font-size: 0.9rem;
}

.info-banner-text {
    color: #788194;
    font-size: 0.75rem;
    line-height: 1.6;
    margin-top: 5px;
}


/* -------------------------------------------------------------------------
   FUTURE SCOPE
------------------------------------------------------------------------- */

.future-card {
    background: linear-gradient(135deg, #171D4A, #312A72);
    color: white;
    border-radius: 22px;
    padding: 1.5rem;
    min-height: 155px;
    box-shadow: 0 20px 45px rgba(31,29,94,0.22);
}

.future-icon {
    font-size: 23px;
    margin-bottom: 10px;
}

.future-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
}

.future-text {
    color: rgba(255,255,255,0.72);
    font-size: 0.71rem;
    line-height: 1.5;
    margin-top: 5px;
}


/* -------------------------------------------------------------------------
   EMPTY STATE
------------------------------------------------------------------------- */

.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    background: white;
    border: 1px dashed #DCDDE7;
    border-radius: 20px;
}

.empty-icon {
    font-size: 42px;
    margin-bottom: 10px;
}

.empty-title {
    font-family: 'Poppins', sans-serif;
    color: #30374A;
    font-weight: 700;
}

.empty-text {
    color: #8A91A4;
    font-size: 0.78rem;
}


/* -------------------------------------------------------------------------
   STREAMLIT INPUTS
------------------------------------------------------------------------- */

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 13px !important;
    border: 1px solid #E2E4EC !important;
    background: white !important;
    transition: 0.2s ease;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    border-color: #8B7CF6 !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,0.10) !important;
}

div.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    border: 1px solid #E2E4EC !important;
    transition: all 0.2s ease !important;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(108,92,231,0.14);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #6547E7, #8D6CF4) !important;
    border: none !important;
    color: white !important;
}


/* -------------------------------------------------------------------------
   FILE UPLOADER
------------------------------------------------------------------------- */

[data-testid="stFileUploader"] {
    background: #FAFAFE;
    border-radius: 14px;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #D7D3F5 !important;
    border-radius: 14px !important;
}


/* -------------------------------------------------------------------------
   EXPANDERS
------------------------------------------------------------------------- */

[data-testid="stExpander"] {
    border: 1px solid #E9EAF2 !important;
    border-radius: 14px !important;
    background: white !important;
}


/* -------------------------------------------------------------------------
   ANIMATIONS
------------------------------------------------------------------------- */

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(-8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# GEMINI CONNECTION
# ============================================================================

if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "⚠️ Gemini API key not found. Add GEMINI_API_KEY in "
        "Streamlit → Settings → Secrets."
    )
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash-lite"


# ============================================================================
# SESSION STATE
# ============================================================================

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


# ============================================================================
# CORE RAG PIPELINE
# ============================================================================

def extract_text_from_pdf(file):
    reader = PdfReader(file)

    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        pages.append(
            {
                "page": i + 1,
                "text": text,
            }
        )

    return pages


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        chunk = " ".join(
            words[start:start + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def embed_text(
    text,
    task_type="RETRIEVAL_DOCUMENT"
):

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type
        ),
    )

    return result.embeddings[0].values


def build_knowledge_base(
    uploaded_files,
    progress_callback=None
):

    knowledge_chunks = []

    for file in uploaded_files:

        pages = extract_text_from_pdf(file)

        for page in pages:

            page_number = page["page"]
            page_text = page["text"]

            if not page_text.strip():
                continue

            chunks = chunk_text(page_text)

            for chunk in chunks:

                knowledge_chunks.append(
                    {
                        "source": file.name,
                        "page": page_number,
                        "text": chunk,
                    }
                )

    if not knowledge_chunks:
        raise ValueError(
            "No readable text was found in the uploaded PDF files. "
            "The PDF may be image/scanned based. OCR may be required."
        )

    embeddings = []

    total = len(knowledge_chunks)

    for i, chunk in enumerate(knowledge_chunks):

        embeddings.append(
            embed_text(chunk["text"])
        )

        if progress_callback:

            progress_callback(
                (i + 1) / total,
                f"Embedding chunk {i + 1}/{total}..."
            )

        if i % 20 == 0 and i > 0:
            time.sleep(1)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return knowledge_chunks, index


def retrieve_top_chunks(
    query,
    index,
    knowledge_chunks,
    top_k=4
):

    query_vec = embed_text(
        query,
        task_type="RETRIEVAL_QUERY"
    )

    query_vec = np.array(
        [query_vec]
    ).astype("float32")

    top_k = min(
        top_k,
        len(knowledge_chunks)
    )

    distances, indices = index.search(
        query_vec,
        top_k
    )

    results = []

    for idx in indices[0]:

        if idx >= 0:

            results.append(
                knowledge_chunks[idx]
            )

    return results


def generate_answer(
    query,
    retrieved_chunks
):

    context_text = "\n\n".join(
        f"""
[Source: {chunk['source']} | Page: {chunk['page']}]

{chunk['text']}
"""
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a corporate knowledge base AI assistant.

Your job is to answer the user's question ONLY using
the retrieved document context.

Rules:

1. Do not use outside knowledge.
2. Do not guess.
3. If the information is not available, clearly say:
   "I couldn't find this information in the uploaded knowledge base."
4. Keep the answer concise and easy to understand.
5. Always mention the relevant source filename and page.
6. Never invent a policy, number, date or procedure.

Retrieved Context:

{context_text}

User Question:

{query}

Generate a professional, natural-language answer.
"""

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )

    return response.text


def rag_query_safe(
    query,
    index,
    knowledge_chunks,
    top_k=4,
    max_retries=3
):

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

                st.toast(
                    f"Rate limit reached. Retrying... "
                    f"{attempt + 1}/{max_retries}"
                )

                time.sleep(20)

            else:

                raise

    raise RuntimeError(
        "Gemini request failed after multiple retries."
    )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                ✨
            </div>

            <div>
                <div class="sidebar-brand-title">
                    KnowledgeHub
                </div>

                <div class="sidebar-brand-sub">
                    AI Knowledge Assistant
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation

    st.markdown("### Workspace")

    nav_items = [
        ("🏠", "Home"),
        ("🤖", "Ask AI"),
        ("📚", "Knowledge Base"),
        ("🔎", "Sources"),
        ("📊", "Analytics"),
        ("🧠", "How RAG Works"),
    ]

    for icon, label in nav_items:

        if st.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
        ):

            st.session_state.active_page = label

    st.markdown("---")

    # Upload

    st.markdown("### 📥 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload company PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload policy documents, handbooks, SOPs and other company knowledge.",
    )

    if st.button(
        "🚀  Build Knowledge Base",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):

        progress_bar = st.progress(
            0.0,
            text="Preparing documents..."
        )

        def update_progress(
            percentage,
            message
        ):

            progress_bar.progress(
                percentage,
                text=message
            )

        try:

            with st.spinner(
                "Building your AI knowledge base..."
            ):

                chunks, index = build_knowledge_base(
                    uploaded_files,
                    progress_callback=update_progress
                )

            st.session_state.knowledge_chunks = chunks
            st.session_state.index = index
            st.session_state.kb_ready = True

            st.session_state.uploaded_documents = [
                file.name
                for file in uploaded_files
            ]

            progress_bar.empty()

            st.success(
                "Knowledge base built successfully!"
            )

            st.balloons()

        except Exception as error:

            progress_bar.empty()

            st.error(
                f"Unable to build knowledge base: {error}"
            )

    # KB status

    if st.session_state.kb_ready:

        st.markdown(
            f"""
            <div class="kb-status kb-ready">
                🟢 Knowledge Base Ready<br>
                <small>
                    {len(st.session_state.knowledge_chunks)}
                    searchable chunks
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="kb-status kb-empty">
                🟡 Knowledge Base not built<br>
                <small>
                    Upload PDFs and build your knowledge base
                </small>
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
            <div style="
                font-weight:700;
                font-size:12px;
                color:#343A55;
            ">
                🎓 MBA462B · CIA-3
            </div>

            <div style="
                font-size:10px;
                color:#858CA0;
                margin-top:3px;
            ">
                Corporate Knowledge Base RAG
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# TOP HEADER
# ============================================================================

st.markdown(
    """
    <div class="top-header">

        <div>
            <div class="greeting-title">
                Good evening 👋
            </div>

            <div class="greeting-sub">
                Your AI-powered organizational knowledge assistant is ready.
            </div>
        </div>

        <div style="
            font-size:26px;
            background:white;
            border:1px solid #ECEEF5;
            border-radius:14px;
            padding:8px 13px;
            box-shadow:0 5px 20px rgba(0,0,0,.04);
        ">
            🔔
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# HERO
# ============================================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-content">

            <div class="hero-eyebrow">
                ✨ AI-POWERED KNOWLEDGE PLATFORM
            </div>

            <h1>
                Corporate Knowledge<br>
                Base Assistant
            </h1>

            <p>
                Ask questions in natural language and get intelligent,
                document-grounded answers from your organization's
                trusted knowledge base.
            </p>

            <div class="badge-row">

                <span class="hero-badge">
                    🔍 Retrieval-Augmented Generation
                </span>

                <span class="hero-badge">
                    📎 Source-Cited Answers
                </span>

                <span class="hero-badge">
                    ⚡ Powered by Gemini
                </span>

                <span class="hero-badge">
                    🔐 Document Grounded
                </span>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# HOME PAGE
# ============================================================================

def render_home():

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    document_count = len(
        st.session_state.uploaded_documents
    )

    chunk_count = len(
        st.session_state.knowledge_chunks
    )

    question_count = len(
        st.session_state.history
    )

    source_count = len(
        set(
            chunk["source"]
            for chunk in st.session_state.knowledge_chunks
        )
    )

    st.markdown(
        """
        <div class="section-header">
            <div>
                <div class="section-title">
                    Knowledge Overview
                </div>

                <div class="section-subtitle">
                    Your organization's AI knowledge ecosystem
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)

    stats = [
        (
            "📄",
            document_count,
            "Documents",
            "#F0EDFF",
            "#6C5CE7",
        ),
        (
            "🧩",
            chunk_count,
            "Knowledge Chunks",
            "#EAF8F4",
            "#10B981",
        ),
        (
            "💬",
            question_count,
            "Questions Asked",
            "#FFF5E8",
            "#F59E0B",
        ),
        (
            "🔗",
            source_count,
            "Active Sources",
            "#EDF5FF",
            "#3B82F6",
        ),
    ]

    for column, stat in zip(
        stat_cols,
        stats
    ):

        icon, value, label, bg, color = stat

        with column:

            st.markdown(
                f"""
                <div class="stat-card">

                    <div class="stat-top">

                        <div class="stat-icon"
                             style="
                                background:{bg};
                                color:{color};
                             ">
                            {icon}
                        </div>

                    </div>

                    <div class="stat-value">
                        {value}
                    </div>

                    <div class="stat-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------------
    # Ask AI
    # ------------------------------------------------------------------------

    st.markdown(
        """
        <div class="section-header">
            <div>
                <div class="section-title">
                    Ask your knowledge base
                </div>

                <div class="section-subtitle">
                    Search your trusted company documents using natural language
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ask-panel">

            <div class="ask-title">
                🤖 What would you like to know?
            </div>

            <div class="ask-subtitle">
                The assistant retrieves relevant information before generating
                an answer.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    q_col, button_col = st.columns(
        [5, 1]
    )

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

        with st.spinner(
            "Searching your knowledge base..."
        ):

            answer, sources = rag_query_safe(
                home_question,
                st.session_state.index,
                st.session_state.knowledge_chunks,
            )

        st.session_state.history.append(
            (
                home_question,
                answer,
                sources,
            )
        )

        st.session_state.active_page = "Ask AI"

        st.rerun()

    st.markdown(
        """
        <div class="try-label">
            Try asking
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "What is the company's maternity leave policy?",
        "How many casual leaves are employees entitled to?",
        "What is the employee onboarding process?",
        "How can I request IT access?",
    ]

    suggestion_cols = st.columns(4)

    for col, suggestion in zip(
        suggestion_cols,
        suggestions
    ):

        with col:

            if st.button(
                suggestion,
                key=f"suggestion_{suggestion}",
                use_container_width=True,
                disabled=not st.session_state.kb_ready,
            ):

                with st.spinner(
                    "Searching documents..."
                ):

                    answer, sources = rag_query_safe(
                        suggestion,
                        st.session_state.index,
                        st.session_state.knowledge_chunks,
                    )

                st.session_state.history.append(
                    (
                        suggestion,
                        answer,
                        sources,
                    )
                )

                st.session_state.active_page = "Ask AI"

                st.rerun()

    # ------------------------------------------------------------------------
    # Feature cards
    # ------------------------------------------------------------------------

    st.markdown(
        """
        <div class="section-header">
            <div>
                <div class="section-title">
                    Intelligent by design
                </div>

                <div class="section-subtitle">
                    Built around the core principles of your RAG project
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_cols = st.columns(4)

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
            "Answers are grounded in your uploaded documents with transparent source references.",
            "#EAF8F4",
            "#10B981",
        ),
        (
            "🔐",
            "Document Grounded",
            "The assistant is instructed not to guess when information cannot be found in the knowledge base.",
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

    for col, feature in zip(
        feature_cols,
        features
    ):

        icon, title, text, bg, color = feature

        with col:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon"
                         style="
                            background:{bg};
                            color:{color};
                         ">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------------
    # Recent Questions
    # ------------------------------------------------------------------------

    if st.session_state.history:

        st.markdown(
            """
            <div class="section-header">
                <div>
                    <div class="section-title">
                        Recent Questions
                    </div>

                    <div class="section-subtitle">
                        Your latest knowledge-base interactions
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for question, answer, sources in reversed(
            st.session_state.history[-5:]
        ):

            source_count = len(sources)

            st.markdown(
                f"""
                <div class="recent-card">

                    <div class="recent-question">
                        💬 {html.escape(question)}
                    </div>

                    <div class="recent-meta">
                        {source_count} source(s) retrieved
                    </div>

                </div>

                <div style="height:6px"></div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# ASK AI PAGE
# ============================================================================

def render_ask_ai():

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    🤖 Ask AI
                </div>

                <div class="section-subtitle">
                    Get answers grounded in your organization's documents
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.kb_ready:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">
                    📚
                </div>

                <div class="empty-title">
                    Your knowledge base is empty
                </div>

                <div class="empty-text">
                    Upload company PDFs from the sidebar and build
                    your knowledge base to start asking questions.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    question = st.text_area(
        "Ask your question",
        placeholder=(
            "Ask anything about your uploaded company documents..."
        ),
        height=100,
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
            (
                question,
                answer,
                sources,
            )
        )

    if st.session_state.history:

        latest_question, latest_answer, latest_sources = (
            st.session_state.history[-1]
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="answer-wrapper">

                <div class="answer-header">

                    <div class="ai-avatar">
                        ✨
                    </div>

                    <div>

                        <div class="answer-label">
                            KnowledgeHub AI
                        </div>

                        <div class="answer-status">
                            ● Grounded response
                        </div>

                    </div>

                </div>

                <div class="answer-question">
                    Q · {html.escape(latest_question)}
                </div>

                <div class="answer-text">
                    {html.escape(latest_answer).replace(chr(10), '<br>')}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-header">

                <div>

                    <div class="section-title">
                        📎 Sources used
                    </div>

                    <div class="section-subtitle">
                        Retrieved document passages supporting the answer
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        source_cols = st.columns(
            min(
                len(latest_sources),
                3
            )
        )

        for col, source in zip(
            source_cols,
            latest_sources
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="source-card">

                        <div class="source-name">
                            📄 {html.escape(source["source"])}
                        </div>

                        <div class="source-meta">
                            Page {source["page"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander(
            "🔎 View retrieved source passages"
        ):

            for source in latest_sources:

                st.markdown(
                    f"""
                    **{source["source"]} — Page {source["page"]}**
                    """
                )

                st.write(
                    source["text"]
                )

                st.divider()


# ============================================================================
# KNOWLEDGE BASE PAGE
# ============================================================================

def render_knowledge_base():

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    📚 Knowledge Base
                </div>

                <div class="section-subtitle">
                    Manage and explore the documents powering your assistant
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.kb_ready:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">
                    📄
                </div>

                <div class="empty-title">
                    No documents indexed yet
                </div>

                <div class="empty-text">
                    Upload PDFs using the sidebar to create your knowledge base.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    documents = st.session_state.uploaded_documents

    cols = st.columns(3)

    for col, document in zip(
        cols,
        documents
    ):

        with col:

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

            st.markdown(
                f"""
                <div class="document-card">

                    <div class="document-icon">
                        📄
                    </div>

                    <div class="document-name">
                        {html.escape(document)}
                    </div>

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

                <div style="height:10px"></div>
                """,
                unsafe_allow_html=True,
            )

    # Categories from the project presentation

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    Knowledge Categories
                </div>

                <div class="section-subtitle">
                    Suggested organizational structure from the project design
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = [
        ("👥", "HR & Operations", "Policies, onboarding, benefits and employee support"),
        ("💻", "IT Knowledge", "Troubleshooting, access requests and IT policies"),
        ("💰", "Finance", "Expenses, reimbursement, budget and compliance"),
        ("🤝", "Sales", "Product pricing, proposals and competitor intelligence"),
        ("📖", "Handbooks", "Employee handbooks and organizational guidance"),
        ("⚙️", "SOPs", "Standard operating procedures and process documentation"),
    ]

    category_cols = st.columns(3)

    for col, category in zip(
        category_cols * 2,
        categories
    ):

        icon, title, description = category

        with col:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon"
                         style="
                            background:#F1F0FF;
                            color:#6C5CE7;
                         ">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-text">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# SOURCES PAGE
# ============================================================================

def render_sources():

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    🔎 Source Management
                </div>

                <div class="section-subtitle">
                    See where your AI knowledge comes from
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.kb_ready:

        st.info(
            "Upload and index documents to view sources."
        )

        return

    sources = {}

    for chunk in st.session_state.knowledge_chunks:

        source_name = chunk["source"]

        if source_name not in sources:
            sources[source_name] = []

        sources[source_name].append(
            chunk
        )

    for source_name, chunks in sources.items():

        pages = sorted(
            set(
                chunk["page"]
                for chunk in chunks
            )
        )

        with st.expander(
            f"📄 {source_name} · {len(pages)} pages"
        ):

            st.write(
                f"**Indexed chunks:** {len(chunks)}"
            )

            st.write(
                f"**Pages represented:** {', '.join(map(str, pages))}"
            )

            for chunk in chunks[:3]:

                st.markdown(
                    f"""
                    **Page {chunk["page"]}**
                    """
                )

                st.caption(
                    chunk["text"][:500]
                    + (
                        "..."
                        if len(chunk["text"]) > 500
                        else ""
                    )
                )


# ============================================================================
# ANALYTICS PAGE
# ============================================================================

def render_analytics():

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    📊 Knowledge Analytics
                </div>

                <div class="section-subtitle">
                    Understand how your knowledge assistant is being used
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    questions = st.session_state.history

    metric_cols = st.columns(4)

    analytics = [
        (
            len(questions),
            "Total Questions",
        ),
        (
            len(
                st.session_state.knowledge_chunks
            ),
            "Indexed Chunks",
        ),
        (
            len(
                st.session_state.uploaded_documents
            ),
            "Documents",
        ),
        (
            len(
                set(
                    c["source"]
                    for c in st.session_state.knowledge_chunks
                )
            ),
            "Active Sources",
        ),
    ]

    for col, (value, label) in zip(
        metric_cols,
        analytics
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat-card">

                    <div class="stat-value">
                        {value}
                    </div>

                    <div class="stat-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    Retrieval Activity
                </div>

                <div class="section-subtitle">
                    Questions and source retrieval activity during this session
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not questions:

        st.info(
            "Ask some questions to populate analytics."
        )

        return

    for index, (
        question,
        answer,
        sources
    ) in enumerate(
        reversed(questions),
        start=1
    ):

        st.markdown(
            f"""
            <div class="recent-card">

                <div class="recent-question">
                    {index}. {html.escape(question)}
                </div>

                <div class="recent-meta">
                    {len(sources)} source(s) retrieved
                </div>

            </div>

            <div style="height:7px"></div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# HOW RAG WORKS
# ============================================================================

def render_rag():

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    🧠 How RAG Works
                </div>

                <div class="section-subtitle">
                    From employee question to grounded AI answer
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-banner">

            <div class="info-banner-title">
                Retrieval-Augmented Generation
            </div>

            <div class="info-banner-text">
                Instead of relying only on the language model's
                pre-trained knowledge, the system retrieves relevant
                information from the organization's documents and uses
                that context to generate the answer.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    workflows = [
        (
            "1",
            "Employee asks",
            "The user asks a question in natural language.",
        ),
        (
            "2",
            "Semantic search",
            "The question is converted into an embedding and compared with document embeddings.",
        ),
        (
            "3",
            "Retrieve chunks",
            "FAISS returns the most relevant knowledge chunks.",
        ),
        (
            "4",
            "Send context to LLM",
            "Relevant document content is provided to Gemini.",
        ),
        (
            "5",
            "Generate answer",
            "Gemini creates a concise natural-language response.",
        ),
        (
            "6",
            "Return sources",
            "The answer is shown together with source documents and page references.",
        ),
    ]

    workflow_cols = st.columns(3)

    for col, workflow in zip(
        workflow_cols * 2,
        workflows
    ):

        number, title, description = workflow

        with col:

            st.markdown(
                f"""
                <div class="workflow-card">

                    <div class="workflow-number">
                        {number}
                    </div>

                    <div class="workflow-title">
                        {title}
                    </div>

                    <div class="workflow-text">
                        {description}
                    </div>

                </div>

                <div style="height:10px"></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    Technology Stack
                </div>

                <div class="section-subtitle">
                    Technologies used in the project architecture
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    tech_cols = st.columns(6)

    technologies = [
        ("💬", "NLP"),
        ("🔎", "Semantic Search"),
        ("🧩", "FAISS"),
        ("🧠", "RAG"),
        ("⚡", "Gemini"),
        ("📄", "PDF Knowledge"),
    ]

    for col, tech in zip(
        tech_cols,
        technologies
    ):

        icon, name = tech

        with col:

            st.markdown(
                f"""
                <div class="feature-card"
                     style="min-height:115px;text-align:center;">

                    <div style="font-size:25px;">
                        {icon}
                    </div>

                    <div class="feature-title"
                         style="margin-top:8px;">
                        {name}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    🚀 Future Scope
                </div>

                <div class="section-subtitle">
                    Expansion opportunities identified in the project
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    future = [
        (
            "🎙️",
            "Voice AI",
            "Ask questions and receive answers using voice.",
        ),
        (
            "🌍",
            "Multi-language",
            "Support multiple languages for diverse users.",
        ),
        (
            "💬",
            "Teams & Slack",
            "Bring knowledge assistance into collaboration platforms.",
        ),
        (
            "👤",
            "Personalized Accounts",
            "Personalized experiences and query history.",
        ),
        (
            "📱",
            "Mobile Application",
            "Access organizational knowledge from anywhere.",
        ),
        (
            "🔄",
            "Real-time Sync",
            "Keep answers aligned with updated documents.",
        ),
    ]

    future_cols = st.columns(3)

    for col, item in zip(
        future_cols * 2,
        future
    ):

        icon, title, description = item

        with col:

            st.markdown(
                f"""
                <div class="future-card">

                    <div class="future-icon">
                        {icon}
                    </div>

                    <div class="future-title">
                        {title}
                    </div>

                    <div class="future-text">
                        {description}
                    </div>

                </div>

                <div style="height:10px"></div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# ROUTING
# ============================================================================

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


# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
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
    """,
    unsafe_allow_html=True,
)
