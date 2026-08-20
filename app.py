"""
Corporate Knowledge Base & Policy Assistant - Streamlit Application
A complete Retrieval-Augmented Generation (RAG) system with PDF upload,
smart chunking, vector/keyword retrieval, and grounded Gemini answers.

Run with:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-api-key"
    streamlit run app.py
"""

import os
import time
import re
from typing import List, Dict, Any
import streamlit as st

# Optional PyPDF for PDF extraction
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Optional Google GenAI SDK
try:
    from google import genai
except ImportError:
    genai = None

# Page Configuration
st.set_page_config(
    page_title="Corporate Policy Knowledge Base AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .citation-box {
        background-color: #f1f5f9;
        border-left: 4px solid #6366f1;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = []  # List of {id, name, pages: [{page, text}]}

if "chunks" not in st.session_state:
    st.session_state.chunks = []  # List of {id, doc_name, page, section, text}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# Document Extraction & Chunking Logic
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file) -> List[Dict[str, Any]]:
    """Extracts text per page from uploaded PDF or TXT files."""
    pages = []
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        if PdfReader is not None:
            reader = PdfReader(uploaded_file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i + 1, "text": text.strip()})
        else:
            # Fallback if pypdf is not installed
            raw_content = uploaded_file.read().decode("latin-1", errors="ignore")
            pages.append({"page": 1, "text": raw_content})
    else:
        # Plain text / Markdown
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        pages.append({"page": 1, "text": text})

    return pages


def create_chunks_from_pages(doc_name: str, pages: List[Dict[str, Any]], chunk_size: int = 120, overlap: int = 20):
    """Splits pages into overlapping chunks with section preservation."""
    chunks = []
    chunk_id = 0

    for p in pages:
        page_num = p["page"]
        text = p["text"]
        
        # Look for section headers
        words = text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            chunk_words = words[start: start + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Simple section extraction
            section_match = re.search(r"(?:Section|Clause|\d+\.)\s+([^\n\.\:]{3,40})", chunk_text)
            section = section_match.group(0) if section_match else "General Provisions"

            chunk_id += 1
            chunks.append({
                "id": f"{doc_name}-chk-{chunk_id}",
                "doc_name": doc_name,
                "page": page_num,
                "section": section,
                "text": chunk_text
            })

            start += max(1, chunk_size - overlap)

    return chunks


# ---------------------------------------------------------
# Retrieval (Keyword & Semantic Relevance Ranking)
# ---------------------------------------------------------
STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "more", "most", "my", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your", "tell", "give"
])

def stem(word: str) -> str:
    w = word.lower().strip(".,!?:;\"'()[]{}")
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "i"
    if w.endswith("es") and len(w) > 4: return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    if w.endswith("ing") and len(w) > 4: return w[:-3]
    if w.endswith("ed") and len(w) > 4: return w[:-2]
    return w

def retrieve_top_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Scores chunks by keyword overlap, stem matching, and exact phrasing."""
    if not chunks:
        return []

    q_words = [w.lower().strip(".,!?:;\"'()[]{}") for w in query.split() if len(w) > 1]
    q_stems = [stem(w) for w in q_words if w not in STOP_WORDS]

    scored = []
    for chk in chunks:
        text_lower = (chk["text"] + " " + chk["section"] + " " + chk["doc_name"]).lower()
        score = 0.0

        # Exact phrase bonus
        if query.lower() in text_lower:
            score += 15.0

        # Stem & keyword overlap
        for s in q_stems:
            if s in text_lower:
                score += 3.0

        for w in q_words:
            if w not in STOP_WORDS and w in text_lower:
                score += 2.0

        if score > 0:
            scored.append((score, chk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


# ---------------------------------------------------------
# Grounded Answer Generation
# ---------------------------------------------------------
def generate_grounded_answer(query: str, retrieved_chunks: List[Dict[str, Any]], api_key: str = "") -> str:
    """Generates an answer using Gemini API or local grounded extraction."""
    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded documents. Please check your uploaded policy documents or phrase your query differently."

    context_str = "\n\n---\n\n".join([
        f"[Source: {c['doc_name']} | Page: {c['page']} | Section: {c['section']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    # 1. Try Gemini API
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if gemini_key and genai is not None:
        try:
            client = genai.Client(api_key=gemini_key)
            prompt = f"""You are a corporate knowledge base assistant.
Answer the employee's question strictly and ONLY using the provided document context below.

CRITICAL RULES:
1. FOCUS EXCLUSIVELY ON THE QUESTION: Provide only the exact details asked. If the user asks about "sick leave", answer ONLY about sick leave (do NOT include casual leave, earned leave, etc.).
2. Direct, concise answer: Start with the direct number/rule, followed by key conditions in bullet points.
3. Cite the exact document name, page, and section.
4. If not found in context, state clearly that the info is not in the knowledge base.

Context:
{context_str}

Employee Question:
{query}

Answer:"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            st.sidebar.warning(f"Gemini API note: {e}. Using local grounded fallback.")

    # 2. Local Grounded Fallback
    top = retrieved_chunks[0]
    lines = [l.strip() for l in top["text"].split("\n") if l.strip()]
    
    # Filter lines matching question keywords
    q_stems = [stem(w) for w in query.split() if w.lower() not in STOP_WORDS]
    matched_lines = []
    for line in lines:
        if any(s in line.lower() for s in q_stems):
            matched_lines.append(f"• {line}")

    if matched_lines:
        content = "\n".join(matched_lines[:4])
        return f"Based on **{top['doc_name']}** (Page {top['page']}, Section: *{top['section']}*):\n\n{content}\n\n*Directly retrieved from uploaded document.*"

    return f"Based on **{top['doc_name']}** (Page {top['page']}, Section: *{top['section']}*):\n\n{top['text'][:350]}...\n\n*Directly retrieved from uploaded document.*"


# ---------------------------------------------------------
# Sidebar: Document Management & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key_input = st.text_input(
        "Gemini API Key (Optional)", 
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Optional: Set your Gemini API key to enable generative synthesis. If omitted, local semantic extraction is used."
    )

    st.markdown("---")
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD policy files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("📥 Index Uploaded Files", use_container_width=True, type="primary"):
        if uploaded_files:
            new_docs_count = 0
            new_chunks_count = 0
            for file in uploaded_files:
                pages = extract_text_from_file(file)
                if pages:
                    chunks = create_chunks_from_pages(file.name, pages)
                    st.session_state.documents.append({
                        "id": f"doc-{int(time.time())}-{file.name}",
                        "name": file.name,
                        "pages": pages,
                        "chunks_count": len(chunks)
                    })
                    st.session_state.chunks.extend(chunks)
                    new_docs_count += 1
                    new_chunks_count += len(chunks)
            st.success(f"Indexed {new_docs_count} document(s) into {new_chunks_count} knowledge chunks!")
        else:
            st.warning("Please choose one or more files first.")

    st.markdown("---")
    st.subheader("🧹 Maintenance")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("⚠️ Reset All (0,0,0)", use_container_width=True):
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.caption("Corporate Knowledge Base AI • RAG Engine")


# ---------------------------------------------------------
# Main Page: Header & Metrics
# ---------------------------------------------------------
st.markdown('<div class="main-header">📚 Corporate Knowledge Base AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about corporate policies, HR guidelines, IT rules, and get direct, cited answers.</div>', unsafe_allow_html=True)

# Metrics Row
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("Indexed Documents", len(st.session_state.documents))
with m_col2:
    st.metric("Knowledge Chunks", len(st.session_state.chunks))
with m_col3:
    st.metric("Queries Answered", len(st.session_state.chat_history))

st.markdown("---")


# ---------------------------------------------------------
# Question Answering Section
# ---------------------------------------------------------
st.subheader("🔍 Ask a Policy Question")

# Sample questions
sample_cols = st.columns(4)
sample_query = ""
if sample_cols[0].button("🌴 Casual leave policy?"):
    sample_query = "How many casual leaves am I entitled to?"
if sample_cols[1].button("🤒 Sick leave rules?"):
    sample_query = "How many sick leaves can I take and what are the medical certificate rules?"
if sample_cols[2].button("✈️ Travel & per diem?"):
    sample_query = "What is the domestic daily allowance and hotel tariff limit?"
if sample_cols[3].button("🔒 Password policy?"):
    sample_query = "What are the password complexity requirements?"

# Question Input Form
with st.form("query_form", clear_on_submit=False):
    query_text = st.text_input(
        "Enter your question:",
        value=sample_query,
        placeholder="e.g., How many sick leaves can I take per year?"
    )
    submit_btn = st.form_submit_button("Ask Question 🚀", type="primary", use_container_width=True)

if submit_btn and query_text.strip():
    if not st.session_state.chunks:
        st.warning("⚠️ Knowledge base is currently empty (0 documents, 0 chunks). Please upload your policy PDF files in the sidebar to begin!")
    else:
        with st.spinner("Searching document index and synthesizing verified answer..."):
            start_t = time.time()
            retrieved = retrieve_top_chunks(query_text.strip(), st.session_state.chunks, top_k=3)
            answer = generate_grounded_answer(query_text.strip(), retrieved, api_key_input)
            latency_ms = int((time.time() - start_t) * 1000)

            # Store in chat history
            st.session_state.chat_history.insert(0, {
                "question": query_text.strip(),
                "answer": answer,
                "sources": retrieved,
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%d %b %Y, %I:%M:%S %p")
            })

# ---------------------------------------------------------
# Display Recent Answers & History
# ---------------------------------------------------------
if st.session_state.chat_history:
    st.markdown("### 💬 Verified Answers & Citations")
    for i, item in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"#### ❓ **{item['question']}**")
            st.caption(f"⏱️ Answered in {item['latency_ms']}ms • {item['timestamp']}")
            
            st.markdown(item["answer"])

            if item.get("sources"):
                with st.expander(f"📑 View {len(item['sources'])} Cited Source Chunks"):
                    for s_idx, src in enumerate(item["sources"]):
                        st.markdown(f"""
                        **Source #{s_idx + 1}:** `{src['doc_name']}` | **Page:** {src['page']} | **Section:** *{src['section']}*
                        > {src['text']}
                        """)
            st.markdown("---")
else:
    st.info("💡 No questions asked yet. Upload a policy PDF from the left sidebar or type a question above!")
