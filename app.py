import os
import json
import hashlib
import inspect
import importlib
import re
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Reload local modules dynamically to prevent stale module caching across hot-reloads
import config
import database
import pdf_processor
import rag_engine
import generator
import evaluator
import diagram_engine
import structured_output

importlib.reload(config)
importlib.reload(database)
importlib.reload(pdf_processor)
# Keep rag_engine's process-level Sentence Transformer model cache alive across
# Streamlit reruns. Restart Streamlit after changing retrieval code.
importlib.reload(generator)
importlib.reload(evaluator)
importlib.reload(diagram_engine)
importlib.reload(structured_output)

from config import PROJECT_INFO, MARK_SCHEMES, UPLOADS_DIR, GEMINI_MODELS
from database import (
    init_db, save_document_record, list_documents,
    save_generated_answer, list_recent_answers,
    save_evaluation, list_recent_evaluations
)
from pdf_processor import AcademicPDFProcessor
from rag_engine import AcademicRAGEngine, RetrievalError, SemanticRetrievalError
from generator import AcademicAnswerGenerator, NOT_ENOUGH_INFO_MSG
from evaluator import StudentAnswerEvaluator
from diagram_engine import DiagramEngine

# Initialize database
init_db()

# Streamlit Page Configuration
st.set_page_config(
    page_title="StudyGenie – AI Academic Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Academic Styling: Soft Blue, Subtle Lavender Accents, White & Slate
st.markdown("""
<style>
    /* Global Theme & Typography */
    :root {
        --primary-blue: #2563eb;
        --primary-blue-hover: #1d4ed8;
        --primary-blue-light: #eff6ff;
        --lavender-accent: #7c3aed;
        --lavender-light: #ede9fe;
        --lavender-tint: #f5f3ff;
        --text-navy: #0f172a;
        --text-slate: #1e293b;
        --text-muted: #64748b;
        --bg-page: #f8fafc;
        --bg-surface: #ffffff;
        --border-light: #e2e8f0;
        --border-dashed: #cbd5e1;
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-sm: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.05);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-page);
        color: var(--text-slate);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--border-light);
        color: var(--text-slate) !important;
    }
    /* Streamlit's native labels use the active theme's text color. Keep them
       readable when the sidebar surface is forced to white above. */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] label * {
        color: var(--text-slate) !important;
    }

    /* Polished Header with Logo */
    .sg-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        background: var(--bg-surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        margin-bottom: 20px;
        box-shadow: none;
    }
    .sg-header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .sg-logo-icon {
        width: 44px;
        height: 44px;
        background: #eff6ff;
        color: #2563eb;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #dbeafe;
        flex-shrink: 0;
    }
    .sg-header-title {
        font-size: 22px;
        font-weight: 700;
        color: var(--text-navy);
        letter-spacing: -0.4px;
        line-height: 1.2;
    }
    .sg-header-subtitle {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 2px;
    }
    /* Welcome Hero Card */
    .sg-welcome-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-light);
        border-top: 3.5px solid #2563eb;
        border-radius: var(--radius-lg);
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: none;
    }
    .sg-welcome-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--lavender-tint);
        color: var(--lavender-accent);
        padding: 4px 11px;
        border-radius: 14px;
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        border: 1px solid var(--lavender-light);
    }
    .sg-welcome-heading {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-navy);
        margin: 0 0 8px 0;
        letter-spacing: -0.4px;
    }
    .sg-welcome-desc {
        font-size: 14.5px;
        color: #475569;
        margin: 0 0 16px 0;
        line-height: 1.6;
        max-width: 860px;
    }
    .sg-feature-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }
    .sg-feature-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12.5px;
        font-weight: 500;
        color: #334155;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 5px 12px;
        border-radius: var(--radius-sm);
    }

    /* Modern Panel Containers */
    .sg-panel {
        background: var(--bg-surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: none;
    }
    .sg-panel:hover {
        border-color: #cbd5e1;
    }

    /* Section Header Rows */
    .sg-section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    .sg-icon-box {
        width: 36px;
        height: 36px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .sg-icon-blue {
        background: var(--primary-blue-light);
        color: var(--primary-blue);
        border: 1px solid #dbeafe;
    }
    .sg-icon-lavender {
        background: var(--lavender-tint);
        color: var(--lavender-accent);
        border: 1px solid var(--lavender-light);
    }
    .sg-section-title {
        font-size: 16.5px;
        font-weight: 600;
        color: var(--text-navy);
        line-height: 1.2;
    }
    .sg-section-subtitle {
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 2px;
    }

    /* Upload Dropzone Container */
    .sg-upload-dropzone {
        border: 2px dashed var(--border-dashed);
        border-radius: var(--radius-md);
        background: #fafbfe;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .sg-upload-dropzone:hover {
        border-color: #3b82f6;
        background: #f0f7ff;
    }

    /* Upload Status Banners */
    .sg-status-active {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        margin-top: 10px;
    }
    .sg-status-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sg-status-title {
        font-size: 13.5px;
        font-weight: 600;
        color: #166534;
    }
    .sg-status-meta {
        font-size: 12.5px;
        color: #15803d;
        margin-top: 2px;
    }
    .sg-status-badge {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }
    .sg-status-idle {
        background: #fafbfe;
        border: 1px dashed var(--border-dashed);
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 13px;
        color: var(--text-muted);
    }

    /* Marks Info Pill */
    .sg-marks-info {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: var(--radius-sm);
        padding: 8px 14px;
        font-size: 13px;
        color: #334155;
        margin-top: 26px;
    }

    /* Answer Display Panel */
    .sg-answer-panel {
        background: #fcfdfe;
        border: 1px solid #e2e8f0;
        border-radius: var(--radius-lg);
        padding: 26px 30px;
        margin-top: 22px;
        box-shadow: none;
    }
    .sg-answer-bar {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 14px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 18px;
    }
    .sg-answer-heading {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-navy);
    }
    .sg-answer-sub {
        font-size: 13.5px;
        color: var(--text-muted);
        margin-top: 3px;
    }
    .sg-answer-tags {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .sg-tag {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .sg-tag-blue {
        background: var(--primary-blue-light);
        color: var(--primary-blue);
        border: 1px solid #bfdbfe;
    }
    .sg-tag-lavender {
        background: var(--lavender-tint);
        color: var(--lavender-accent);
        border: 1px solid var(--lavender-light);
    }

    /* Scorecard for Evaluation */
    .sg-scorecard {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: var(--radius-md);
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: none;
    }
    .sg-score-title {
        font-size: 21px;
        font-weight: 700;
        color: var(--text-navy);
        margin: 0;
    }
    .sg-score-meta {
        font-size: 14px;
        color: #475569;
        margin-top: 4px;
    }

    /* Citations Box */
    .sg-source-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: var(--radius-sm);
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 13px;
    }

    /* Buttons Styling */
    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        padding: 8px 22px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Main sections use a horizontal, keyboard-accessible navigation radio. */
    [data-testid="stRadio"] > label {
        display: none;
    }
    [data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        gap: 4px;
        align-items: stretch;
        border-bottom: 1px solid var(--border-light);
        margin: 0 0 18px;
        width: 100%;
    }
    [data-testid="stRadio"] [role="radiogroup"] label {
        color: #475569;
        font-size: 14px;
        font-weight: 500;
        padding: 10px 18px 9px;
        margin: 0;
        border-radius: 8px 8px 0 0;
        border-bottom: 2px solid transparent;
        cursor: pointer;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:hover {
        color: var(--primary-blue);
        background: #f1f5f9;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        color: var(--primary-blue);
        background: #eff6ff;
        border-bottom-color: var(--primary-blue);
        font-weight: 600;
    }

    @media (max-width: 768px) {
        .sg-header {
            align-items: flex-start;
            flex-direction: column;
            gap: 12px;
            padding: 16px;
        }
        .sg-panel,
        .sg-answer-panel {
            padding: 16px;
        }
        .sg-answer-bar {
            flex-direction: column;
            gap: 12px;
        }
        .sg-status-active {
            align-items: flex-start;
            flex-direction: column;
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 10px;
            font-size: 13px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
def _is_valid_rag_engine(engine):
    if not engine or not hasattr(engine, "search") or not hasattr(engine, "clear"):
        return False
    try:
        sig = inspect.signature(engine.search)
        return "active_doc_ids" in sig.parameters
    except Exception:
        return False


def _configured_api_key():
    """Read a server key from environment or Streamlit secrets without exposing it."""
    key = os.getenv("GEMINI_API_KEY", "")
    secret_files = (
        Path.home() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    )
    # Accessing st.secrets with no secrets.toml configured makes Streamlit emit
    # a confusing "No secrets found" message even though the sidebar key field
    # and offline mode remain available.
    if not any(secret_file.is_file() for secret_file in secret_files):
        return key
    try:
        key = st.secrets.get("GEMINI_API_KEY", key) or key
    except Exception:
        pass
    return key


def _new_rag_engine(retrieval_method="semantic", min_score_threshold=0.25):
    """Create local document retrieval; Gemini keys are only for generation."""
    return AcademicRAGEngine(
        retrieval_method=retrieval_method,
        min_score_threshold=min_score_threshold,
    )


def _sync_rag_engine(retrieval_method="semantic", min_score_threshold=0.25):
    """Rebuild the active-document index when its retrieval method changes."""
    engine = st.session_state.get("rag_engine")
    if (
        engine is None
        or getattr(engine, "retrieval_method", None) != retrieval_method
    ):
        engine = _new_rag_engine(retrieval_method, min_score_threshold)
        if st.session_state.get("all_chunks"):
            engine.build_index(st.session_state.all_chunks)
        st.session_state.rag_engine = engine
    else:
        engine.min_score_threshold = float(min_score_threshold)
    return engine

if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = "semantic"
if "semantic_min_score" not in st.session_state:
    st.session_state.semantic_min_score = 0.25
if (
    "rag_engine" not in st.session_state
    or not _is_valid_rag_engine(st.session_state.rag_engine)
    or getattr(st.session_state.rag_engine, "retrieval_method", None) != st.session_state.retrieval_method
):
    st.session_state.rag_engine = _new_rag_engine(
        st.session_state.retrieval_method,
        st.session_state.semantic_min_score,
    )
    if st.session_state.get("all_chunks"):
        st.session_state.rag_engine.build_index(st.session_state.all_chunks)
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "current_evaluation" not in st.session_state:
    st.session_state.current_evaluation = None
if "selected_question_for_eval" not in st.session_state:
    st.session_state.selected_question_for_eval = ""
if "current_paper_evaluation" not in st.session_state:
    st.session_state.current_paper_evaluation = None
if "user_question_input" not in st.session_state:
    st.session_state.user_question_input = ""
if "target_marks_input" not in st.session_state:
    st.session_state.target_marks_input = 10
if "active_doc_ids" not in st.session_state:
    st.session_state.active_doc_ids = []

# Document Ingestion Helper
def process_uploaded_files(files, replace=True):
    if not files:
        raise ValueError("Choose at least one PDF or text file to process.")

    processor = AcademicPDFProcessor()
    new_chunks = []
    new_metadata = []
    created_paths = []
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded_file in files:
            original_name = Path(uploaded_file.name).name
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name).strip("._")
            suffix = Path(safe_name).suffix.lower()
            if suffix not in {".pdf", ".txt"}:
                raise ValueError(f"Unsupported file type for '{original_name}'. Upload a PDF or TXT file.")

            content = bytes(uploaded_file.getbuffer())
            if not content:
                raise ValueError(f"'{safe_name}' is empty.")
            if suffix == ".pdf" and b"%PDF-" not in content[:1024]:
                raise ValueError(f"'{safe_name}' is not a valid PDF file.")

            content_hash = hashlib.sha256(content).hexdigest()[:12]
            save_path = UPLOADS_DIR / f"{content_hash}_{safe_name}"
            if not save_path.exists():
                save_path.write_bytes(content)
                created_paths.append(save_path)

            info = processor.process_file(str(save_path), safe_name)
            if not info["chunks"]:
                raise ValueError(f"No readable text was found in '{safe_name}'. Try a text-based PDF or TXT file.")
            new_chunks.extend(info["chunks"])
            new_metadata.append(info)

        existing_chunks = [] if replace else list(st.session_state.all_chunks)
        existing_metadata = [] if replace else list(st.session_state.doc_metadata)
        candidate_chunks = existing_chunks + new_chunks
        candidate_metadata = existing_metadata + new_metadata

        # Build the replacement index before changing active session state.
        retrieval_method = st.session_state.get("retrieval_method", "semantic")
        min_score_threshold = (
            st.session_state.get("semantic_min_score", 0.25)
            if retrieval_method == "semantic" else 0.05
        )
        candidate_rag = _new_rag_engine(retrieval_method, min_score_threshold)
        candidate_rag.build_index(candidate_chunks)
        for info in new_metadata:
            info["database_id"] = save_document_record(
                filename=info["source_doc"],
                file_path=info["file_path"],
                total_pages=info["total_pages"],
                chunk_count=info["total_chunks"],
                file_size=Path(info["file_path"]).stat().st_size,
            )

        st.session_state.all_chunks = candidate_chunks
        st.session_state.doc_metadata = candidate_metadata
        st.session_state.rag_engine = candidate_rag
        st.session_state.active_doc_ids = [d.get("doc_id") for d in candidate_metadata]
        st.session_state.current_answer = None
        st.session_state.current_evaluation = None
        st.session_state.user_question_input = ""
    except Exception:
        for path in created_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

# Header with Book Logo & Brand
st.markdown("""
<div class="sg-header">
    <div class="sg-header-left">
        <div class="sg-logo-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"></path>
                <path d="M6 6h10"></path>
                <path d="M6 10h10"></path>
                <path d="M6 14h6"></path>
            </svg>
        </div>
        <div>
            <div class="sg-header-title">StudyGenie</div>
            <div class="sg-header-subtitle">AI Academic Assistant</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("### Settings & Controls")
    
    st.markdown("##### AI Model Tier")
    retrieval_labels = {
        "Semantic Search (local Sentence Transformer)": "semantic",
        "TF-IDF Keyword Search (local)": "tfidf",
    }
    selected_retrieval_label = st.selectbox(
        "Document Retrieval Method",
        options=list(retrieval_labels),
        index=0 if st.session_state.retrieval_method == "semantic" else 1,
        help="Semantic search uses a local Sentence Transformer model. TF-IDF is an independent keyword-search option.",
    )
    selected_retrieval_method = retrieval_labels[selected_retrieval_label]
    st.session_state.retrieval_method = selected_retrieval_method
    if selected_retrieval_method == "semantic":
        semantic_threshold = st.slider(
            "Minimum cosine similarity",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.semantic_min_score),
            step=0.05,
            help="Filters semantic results by cosine similarity. This score is not a probability.",
        )
        st.session_state.semantic_min_score = semantic_threshold
    else:
        semantic_threshold = 0.05

    active_api_key = _configured_api_key() or None
    _sync_rag_engine(selected_retrieval_method, semantic_threshold)
    retrieval_status = st.empty()
    retrieval_status.caption(f"Retrieval mode: {st.session_state.rag_engine.retrieval_mode}")
    if st.session_state.rag_engine.retrieval_error and st.session_state.all_chunks:
        st.warning(st.session_state.rag_engine.retrieval_error)
        if selected_retrieval_method == "semantic" and st.button(
            "Retry semantic indexing", use_container_width=True, key="retry_semantic_index"
        ):
            with st.spinner("Retrying local semantic indexing..."):
                st.session_state.rag_engine.build_index(st.session_state.all_chunks)
            st.rerun()

    selected_model_label = st.selectbox(
        "Answer Generation Model",
        options=list(GEMINI_MODELS),
        index=0,
        help="Choose a Gemini model for answer generation and evaluation. PDF retrieval runs locally."
    )
    selected_model = GEMINI_MODELS[selected_model_label]

    st.divider()

    st.markdown("##### Active Documents")
    if st.session_state.doc_metadata:
        doc_names = [d.get("source_doc", "Document") for d in st.session_state.doc_metadata]
        if len(doc_names) > 1:
            scope_options = ["All Selected Documents"] + doc_names
            selected_scope = st.selectbox(
                "Filter retrieval to:",
                options=scope_options,
                index=0
            )
            if selected_scope == "All Selected Documents":
                st.session_state.active_doc_ids = [d.get("doc_id") for d in st.session_state.doc_metadata]
            else:
                st.session_state.active_doc_ids = [
                    d.get("doc_id") for d in st.session_state.doc_metadata 
                    if d.get("source_doc") == selected_scope
                ]
        else:
            st.session_state.active_doc_ids = [st.session_state.doc_metadata[0].get("doc_id")]
            st.caption(f"Active file: **{doc_names[0]}**")
    else:
        st.session_state.active_doc_ids = []
        st.caption("No document loaded.")

    if st.button("Clear Documents", use_container_width=True):
        st.session_state.all_chunks = []
        st.session_state.doc_metadata = []
        st.session_state.current_answer = None
        st.session_state.current_evaluation = None
        st.session_state.user_question_input = ""
        st.session_state.active_doc_ids = []
        if hasattr(st.session_state.rag_engine, "clear"):
            st.session_state.rag_engine.clear()
        else:
            st.session_state.rag_engine = AcademicRAGEngine()
        st.rerun()

# Main Navigation
default_q = st.session_state.get("user_question_input", "")
active_section = st.radio(
    "Main navigation",
    ["Ask Questions", "Evaluate Answer", "Document Explorer", "History"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)

# =========================================================================
# Main Content Sections
# =========================================================================
if active_section == "Ask Questions":
    # Welcome Hero Section
    st.markdown("""
    <div class="sg-welcome-card">
        <div class="sg-welcome-tag">Intelligent Academic Assistant</div>
        <h1 class="sg-welcome-heading">Your notes. Your questions. Clear answers.</h1>
        <p class="sg-welcome-desc">
            Upload your course notes, textbooks, or syllabus. Ask exam questions to receive syllabus-grounded answers calibrated to university mark rubrics (2M, 5M, 10M, 15M), verified architectural diagrams, and structured grading schemes.
        </p>
        <div class="sg-feature-strip">
            <span class="sg-feature-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                PDF-Grounded Retrieval
            </span>
            <span class="sg-feature-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                Points Formula: Marks &times; 2
            </span>
            <span class="sg-feature-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                Mermaid Diagram Generation
            </span>
            <span class="sg-feature-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                SIT / VTU University Grading
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. PDF Upload Section (Dashed Border with Document Icon)
    with st.container():
        st.markdown("""
        <div class="sg-panel">
            <div class="sg-section-header">
                <div class="sg-icon-box sg-icon-blue">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="12" y1="18" x2="12" y2="12"></line>
                        <line x1="9" y1="15" x2="15" y2="15"></line>
                    </svg>
                </div>
                <div>
                    <div class="sg-section-title">Upload Course Material</div>
                    <div class="sg-section-subtitle">Drop your lecture notes, question banks, or textbook chapters in PDF format</div>
                </div>
            </div>
            <div class="sg-upload-dropzone">
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload course notes (PDF)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        col_up1, col_up2 = st.columns([3, 1])
        with col_up1:
            replace_existing = st.checkbox(
                "Replace previously uploaded documents",
                value=True,
                help="When checked, previous notes are cleared when uploading new files."
            )
        with col_up2:
            if uploaded_files:
                if st.button("Process Document", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Extracting text, chunking, and building the document index..."):
                            process_uploaded_files(uploaded_files, replace=replace_existing)
                        st.success(f"Indexed {len(uploaded_files)} file(s) into {len(st.session_state.all_chunks)} chunks.")
                        if st.session_state.rag_engine.retrieval_error:
                            st.warning(st.session_state.rag_engine.retrieval_error)
                        st.rerun()
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
                    except Exception:
                        st.error("Document processing failed. Your previously indexed documents are still active; check the server log and try again.")

        # Uploaded status display
        if st.session_state.doc_metadata:
            doc_names = ", ".join([d.get("source_doc", "Document") for d in st.session_state.doc_metadata])
            total_p = sum([d.get("total_pages", 1) for d in st.session_state.doc_metadata])
            total_c = len(st.session_state.all_chunks)
            st.markdown(f"""
            <div class="sg-status-active">
                <div class="sg-status-left">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <div>
                        <div class="sg-status-title">{doc_names}</div>
                        <div class="sg-status-meta">{total_p} page(s) • {total_c} searchable chunks indexed</div>
                    </div>
                </div>
                <div class="sg-status-badge">RAG Ready</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sg-status-idle">
                No document currently uploaded. Upload a PDF or TXT file above to begin.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Spacious Question Section
    with st.container():
        st.markdown("""
        <div class="sg-panel">
            <div class="sg-section-header">
                <div class="sg-icon-box sg-icon-lavender">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                </div>
                <div>
                    <div class="sg-section-title">Ask an Academic Question</div>
                    <div class="sg-section-subtitle">Type your question or pick directly from your document's question bank</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        default_q = st.session_state.get("user_question_input", "")

        user_question = st.text_area(
            "Question input:",
            value=default_q,
            height=85,
            placeholder="e.g., Explain the concept of paging with hardware address translation and neat diagram.",
            label_visibility="collapsed"
        )

        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            default_marks_idx = 2
            preset_target = st.session_state.get("target_marks_input", 10)
            if preset_target in [2, 5, 10, 15]:
                default_marks_idx = [2, 5, 10, 15].index(preset_target)

            target_marks = st.selectbox(
                "Marks Allotted:",
                options=[2, 5, 10, 15],
                index=default_marks_idx,
                format_func=lambda m: f"{m} Marks ({MARK_SCHEMES[m]['label'].split('(')[1].replace(')', '')})"
            )
        with col_m2:
            scheme_info = MARK_SCHEMES[target_marks]
            expected_pts = target_marks * 2
            st.markdown(f"""
            <div class="sg-marks-info">
                <strong>Expected Points:</strong> {expected_pts} points ({target_marks} Marks &times; 2) &nbsp;|&nbsp; 
                <strong>Standard:</strong> {scheme_info['expected_length']}
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        if st.button("Get Answer", type="primary", use_container_width=True):
            if not user_question.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.all_chunks:
                st.warning("Please upload a PDF document first.")
            else:
                try:
                    with st.spinner(f"Retrieving context from notes and generating {target_marks}-mark answer..."):
                        top_k = 3 if target_marks == 2 else (5 if target_marks == 5 else 7)
                        active_ids = st.session_state.get("active_doc_ids")
                        retrieved_chunks = st.session_state.rag_engine.search(
                            user_question,
                            top_k=top_k,
                            active_doc_ids=active_ids,
                        )

                        generator_inst = AcademicAnswerGenerator(
                            api_key=active_api_key,
                            model_name=selected_model
                        )
                        result = generator_inst.generate_scheme_and_answer(
                            question=user_question,
                            marks=target_marks,
                            context_chunks=retrieved_chunks
                        )

                    st.session_state.current_answer = result
                    st.session_state.selected_question_for_eval = user_question
                    try:
                        primary_doc_id = (
                            st.session_state.doc_metadata[0].get("database_id")
                            if st.session_state.doc_metadata else None
                        )
                        save_generated_answer(
                            doc_id=primary_doc_id,
                            question=user_question,
                            marks=target_marks,
                            answer_text=result["answer_markdown"],
                            scheme_dict=result["scheme"],
                            diagram_code=result.get("diagram_code", ""),
                            sources=result.get("sources", [])
                        )
                    except Exception:
                        st.warning("The answer was generated, but it could not be saved to answer history.")
                except RetrievalError as exc:
                    retrieval_status.caption(f"Retrieval mode: {st.session_state.rag_engine.retrieval_mode}")
                    st.error(str(exc))
                except Exception:
                    st.error("Answer generation failed. Check that the document is processed and the selected model is configured, then try again.")

        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Generated Answer Section (Subtle Tinted Background, Well-Spaced)
    if st.session_state.current_answer:
        ans_data = st.session_state.current_answer
        st.markdown('<div class="sg-answer-panel">', unsafe_allow_html=True)
        
        if ans_data.get("answer_markdown") == NOT_ENOUGH_INFO_MSG:
            st.warning("The uploaded PDF does not contain enough relevant information to answer this question.")
            st.info("The selected retrieval method found no content directly addressing this question in the active document. Please check your question or selected document.")
        else:
            # Header Bar
            st.markdown(f"""
            <div class="sg-answer-bar">
                <div>
                    <div class="sg-answer-heading">Answer & Marking Scheme</div>
                    <div class="sg-answer-sub">Question: {ans_data['question']}</div>
                </div>
                <div class="sg-answer-tags">
                    <span class="sg-tag sg-tag-blue">{ans_data['marks']} Marks Standard</span>
                    <span class="sg-tag sg-tag-lavender">{ans_data['marks'] * 2} Expected Points</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            structured_data = ans_data.get("structured_output")
            if structured_data and ans_data.get("is_structured", True):
                out_tab1, out_tab2, out_tab3 = st.tabs([
                    "Formatted Model Answer",
                    "Structured Breakdown",
                    "Machine JSON Schema"
                ])
                with out_tab1:
                    st.markdown(ans_data["answer_markdown"])
                with out_tab2:
                    summary = structured_data.get("executive_summary")
                    if summary:
                        st.info(f"**Overview:** {summary}")
                    for sec in structured_data.get("sections", []):
                        st.markdown(f"#### {sec.get('heading', 'Section')}")
                        if sec.get("summary"):
                            st.caption(sec["summary"])
                        sec_type = sec.get("type")
                        if sec_type == "table":
                            tdata = sec.get("table_data", {})
                            headers = tdata.get("headers", [])
                            rows = tdata.get("rows", [])
                            if headers and rows:
                                table_md = "| " + " | ".join(headers) + " |\n"
                                table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                                for r in rows:
                                    table_md += "| " + " | ".join([str(cell).replace("|", "/") for cell in r]) + " |\n"
                                st.markdown(table_md)
                        elif sec_type == "key_value":
                            items = sec.get("items", [])
                            for it in items:
                                st.markdown(f"- **{it.get('key', 'Property')}:** {it.get('value', '')}")
                        elif sec_type == "ordered_steps":
                            for s in sec.get("steps", []):
                                st.markdown(f"**Step {s.get('step_number', 1)}: {s.get('phase_name', 'Phase')}** — {s.get('action', '')}")
                        elif sec_type == "bullet_points":
                            for idx_pt, pt in enumerate(sec.get("points", []), 1):
                                st.markdown(f"**{idx_pt}.** {pt}")
                with out_tab3:
                    st.json(structured_data)
            else:
                st.markdown(ans_data["answer_markdown"])

            # Diagram Display
            if ans_data.get("diagram_needed") and ans_data.get("diagram_code"):
                st.divider()
                diag_title = ans_data.get("diagram_title", "Architecture / Workflow Diagram")
                st.markdown(f"#### Diagram: {diag_title}")
                diagram_html = DiagramEngine.generate_html_embed(ans_data["diagram_code"], diag_title)
                components.html(diagram_html, height=360, scrolling=True)
                with st.expander("View Mermaid Diagram Code"):
                    st.code(ans_data["diagram_code"], language="mermaid")
            elif ans_data.get("diagram_message"):
                st.divider()
                st.caption(f"Diagram Note: {ans_data['diagram_message']}")

            # Marking Scheme Table
            st.divider()
            st.markdown(f"#### Evaluation Marking Scheme ({ans_data['marks']} Marks)")
            scheme_obj = ans_data.get("scheme", {})
            components_list = scheme_obj.get("components", [])
            if components_list:
                scheme_md = "| Criterion / Component | Weightage |\n| --- | ---: |\n"
                for component in components_list:
                    criterion = str(component.get("component", "Criterion")).replace("|", "/")
                    weightage = f"{component.get('marks')} Marks"
                    scheme_md += f"| {criterion} | {weightage} |\n"
                st.markdown(scheme_md)

            # Examiner Tips
            if ans_data.get("tips_for_full_marks"):
                with st.expander("Examiner Tips for Scoring Full Marks"):
                    for tip in ans_data["tips_for_full_marks"]:
                        st.markdown(f"- {tip}")

            # Verified Citations
            if ans_data.get("sources"):
                with st.expander(f"Verified Citations from PDF ({len(ans_data['sources'])} chunks)"):
                    for src in ans_data["sources"]:
                        st.markdown(f"""
                        <div class="sg-source-item">
                            <strong>{src.get('source_doc', 'Document')}</strong> — Page {src.get('page', 1)} &nbsp;|&nbsp; {('Cosine similarity' if src.get('score_kind') == 'cosine_similarity' else ('TF-IDF cosine score' if src.get('score_kind') == 'tfidf_cosine_similarity' else 'Similarity score'))}: <code>{src.get('score', 'n/a')}</code><br/>
                            <span style="color: #475569;">"{src.get('text', '')[:220]}..."</span>
                        </div>
                        """, unsafe_allow_html=True)

            # Download Action
            download_content = f"""# {PROJECT_INFO['title']}
## Question: {ans_data['question']} ({ans_data['marks']} Marks)

---
### EVALUATION SCHEME
{json.dumps(ans_data.get('scheme', {}), indent=2)}

---
### MODEL ANSWER
{ans_data['answer_markdown']}

---
### DIAGRAM (MERMAID)
```mermaid
{ans_data.get('diagram_code', '')}
```
"""
            st.write("")
            st.download_button(
                label="Download Answer Sheet (Markdown)",
                data=download_content,
                file_name=f"Answer_{ans_data['marks']}M.md",
                mime="text/markdown"
            )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# Student Answer Evaluator
# =========================================================================
elif active_section == "Evaluate Answer":
    st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sg-section-header">
        <div class="sg-icon-box sg-icon-lavender">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 20h9"></path>
                <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
            </svg>
        </div>
        <div>
            <div class="sg-section-title">Student Answer Evaluator & Grading</div>
            <div class="sg-section-subtitle">Evaluate responses against official syllabus marking schemes and calculate university grades</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ev_mode_tab1, ev_mode_tab2 = st.tabs([
        "Single Question Grading",
        "Full Paper Evaluation"
    ])

    # -------------------------------------------------------------
    # SUBTAB 1: Single Question Grading
    # -------------------------------------------------------------
    with ev_mode_tab1:
        eval_q = st.text_input(
            "Examination Question:",
            value=st.session_state.selected_question_for_eval or (default_q or "Enter question to evaluate"),
            placeholder="Enter question"
        )
        
        col_ev_m1, col_ev_m2 = st.columns([1, 1])
        with col_ev_m1:
            eval_marks = st.selectbox(
                "Target Marks:",
                options=[2, 5, 10, 15],
                index=2,
                key="eval_marks_select"
            )
            expected_pts_guide = eval_marks * 2
            st.caption(f"Expected points guide: **{expected_pts_guide} points** ({eval_marks} Marks × 2)")
        with col_ev_m2:
            st.write("")
            if st.button("Load Sample Answer for Testing", use_container_width=True):
                st.session_state.sample_student_input = (
                    f"Regarding {eval_q}:\n"
                    "This concept provides core functional capability and data management. "
                    "It consists of distinct modular components operating across sequential stages. "
                    "The primary advantages are modular design and predictable latency."
                )

        student_text = st.text_area(
            "Student Submitted Answer:",
            value=st.session_state.get("sample_student_input", ""),
            height=150,
            placeholder="Write or paste the student's answer here..."
        )

        col_run1, col_run2 = st.columns([2, 1])
        with col_run1:
            eval_button = st.button("Evaluate Answer", type="primary", use_container_width=True)
        with col_run2:
            if st.button("Reset Evaluation", use_container_width=True):
                st.session_state.current_evaluation = None
                st.session_state.sample_student_input = ""
                st.rerun()

        if eval_button:
            if not student_text.strip():
                st.warning("Please provide a student answer to evaluate.")
            else:
                with st.spinner("Evaluating response against marking scheme..."):
                    if (st.session_state.current_answer 
                        and st.session_state.current_answer.get("question") == eval_q 
                        and st.session_state.current_answer.get("marks") == eval_marks):
                        active_scheme = st.session_state.current_answer["scheme"]
                        model_ans = st.session_state.current_answer["answer_markdown"]
                    else:
                        active_scheme = None
                        model_ans = ""
                        if st.session_state.get("all_chunks") and hasattr(st.session_state.rag_engine, "search"):
                            try:
                                top_chunks = st.session_state.rag_engine.search(
                                    eval_q,
                                    top_k=4,
                                    active_doc_ids=st.session_state.get("active_doc_ids")
                                )
                                if top_chunks:
                                    gen = AcademicAnswerGenerator(
                                        api_key=active_api_key,
                                        model_name=selected_model
                                    )
                                    ref_obj = gen.generate_scheme_and_answer(eval_q, eval_marks, top_chunks)
                                    if ref_obj.get("has_sufficient_context"):
                                        model_ans = ref_obj.get("answer_markdown", "")
                                        active_scheme = ref_obj.get("scheme")
                            except RetrievalError as rag_err:
                                st.warning(f"Reference retrieval was unavailable: {rag_err}")

                    evaluator = StudentAnswerEvaluator(
                        api_key=active_api_key,
                        model_name=selected_model
                    )
                    
                    eval_res = evaluator.evaluate_answer(
                        question=eval_q,
                        target_marks=eval_marks,
                        student_answer=student_text,
                        scheme=active_scheme,
                        model_answer=model_ans
                    )
                    
                    st.session_state.current_evaluation = eval_res
                    
                    save_evaluation(
                        answer_id=None,
                        question=eval_q,
                        target_marks=eval_marks,
                        marks_awarded=eval_res["marks_awarded"],
                        student_answer=student_text,
                        feedback=eval_res
                    )

        # Render Evaluation Results
        if st.session_state.current_evaluation:
            ev = st.session_state.current_evaluation
            st.divider()

            status_text = "Pass" if ev.get("status") == "Pass" else "Re-attempt Recommended"
            st.markdown(f"""
            <div class="sg-scorecard">
                <div class="sg-score-title">Score: {ev['marks_awarded']} / {ev['target_marks']} Marks ({ev['percentage']}%)</div>
                <div class="sg-score-meta">
                    University Grade: <strong>{ev['grade']}</strong> &nbsp;|&nbsp; Status: <strong>{status_text}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Marks Awarded", f"{ev['marks_awarded']} / {ev['target_marks']} M")
            col_m2.metric("Score Percentage", f"{ev['percentage']}%")
            col_m3.metric("University Grade", ev['grade'].split("(")[0].strip())
            col_m4.metric("Expected Points Guide", f"{ev.get('expected_points', eval_marks*2)} pts")

            # Rubric Criteria Classification
            st.markdown("##### Criteria Classification")
            col_crit1, col_crit2 = st.columns(2)
            with col_crit1:
                st.markdown("**Criteria Satisfied:**")
                if ev.get("criteria_satisfied"):
                    for s in ev["criteria_satisfied"]:
                        st.markdown(f"- {s}")
                else:
                    st.caption("No criteria fully satisfied.")

                st.markdown("**Missing Criteria:**")
                if ev.get("criteria_missing"):
                    for m in ev["criteria_missing"]:
                        st.markdown(f"- {m}")
                else:
                    st.caption("No mandatory criteria omitted.")

            with col_crit2:
                st.markdown("**Partially Satisfied Criteria:**")
                if ev.get("criteria_partially_satisfied"):
                    for p in ev["criteria_partially_satisfied"]:
                        st.markdown(f"- {p}")
                else:
                    st.caption("None.")

                st.markdown("**Incorrect Statements / Misconceptions:**")
                if ev.get("incorrect_statements"):
                    for inc in ev["incorrect_statements"]:
                        if "None detected" in inc:
                            st.caption(inc)
                        else:
                            st.markdown(f"<div style='background:#fef2f2;border:1px solid #fecaca;padding:6px 10px;margin-bottom:6px;border-radius:4px;color:#991b1b;font-size:13px;'>{inc}</div>", unsafe_allow_html=True)
                else:
                    st.caption("None detected.")

            st.divider()

            # Rubric Breakdown Table
            st.markdown(f"##### Rubric Breakdown ({ev.get('target_marks', eval_marks)} Marks)")
            comp_evals = ev.get("component_evaluations", [])
            if comp_evals:
                breakdown_table = []
                for item in comp_evals:
                    breakdown_table.append({
                        "Criterion": item.get("criterion"),
                        "Allocated": f"{item.get('allocated_marks')} M",
                        "Awarded": f"{item.get('marks_awarded')} M",
                        "Status": item.get("status"),
                        "Remarks": item.get("comments")
                    })
                st.table(breakdown_table)

            # Feedback
            st.markdown("##### Examiner Feedback")
            st.info(ev.get("examiner_feedback", "Ensure core technical definitions and diagrams are included."))

            # Human Review & Score Adjustment
            with st.expander("Score Override (Examiner Review)"):
                st.caption("Adjust score with recorded rationale for university records.")
                col_ov1, col_ov2 = st.columns(2)
                with col_ov1:
                    adj_score = st.number_input(
                        f"Adjusted Marks (Max: {ev['target_marks']}):",
                        min_value=0.0,
                        max_value=float(ev['target_marks']),
                        value=float(ev['marks_awarded']),
                        step=0.5,
                        key="adj_score_input"
                    )
                    reviewer = st.text_input("Reviewer Name:", value="Examiner", key="reviewer_name_input")
                with col_ov2:
                    override_reason = st.text_area(
                        "Override Reason:",
                        placeholder="e.g., Awarded discretionary credit for valid steps.",
                        key="override_reason_input"
                    )
                    if st.button("Apply Score Override"):
                        if not override_reason.strip():
                            st.warning("Please enter a reason for the override.")
                        else:
                            evaluator = StudentAnswerEvaluator()
                            updated_ev = evaluator.apply_human_override(
                                ev,
                                adjusted_marks=adj_score,
                                override_reason=override_reason,
                                reviewer_name=reviewer
                            )
                            st.session_state.current_evaluation = updated_ev
                            st.success(f"Score updated to {adj_score}/{ev['target_marks']} Marks.")
                            st.rerun()

            st.write("")
            st.download_button(
                label="Download Grading Report (Markdown)",
                data=ev.get("report_markdown", ""),
                file_name=f"Grading_Report_{ev.get('target_marks')}M.md",
                mime="text/markdown",
                key="dl_single_eval_md"
            )

    # -------------------------------------------------------------
    # SUBTAB 2: Full Paper Evaluation
    # -------------------------------------------------------------
    with ev_mode_tab2:
        st.markdown("##### Multi-Question Paper Evaluation")
        st.caption("Grade multiple questions together to compute total marks, percentage, and overall grade.")

        col_q1_a, col_q1_b = st.columns([3, 1])
        with col_q1_a:
            paper_q1 = st.text_input("Question 1:", value=st.session_state.get("paper_q1_text", "Define paging in operating systems."), key="pq1_text")
        with col_q1_b:
            paper_q1_m = st.selectbox("Q1 Marks:", [2, 5, 10, 15], index=0, key="pq1_marks")
        paper_q1_ans = st.text_area("Q1 Student Answer:", value="Paging is a memory management scheme that divides logical address space into pages and physical memory into frames.", height=65, key="pq1_ans")

        col_q2_a, col_q2_b = st.columns([3, 1])
        with col_q2_a:
            paper_q2 = st.text_input("Question 2:", value=st.session_state.get("paper_q2_text", "Explain the working of hardware address translation."), key="pq2_text")
        with col_q2_b:
            paper_q2_m = st.selectbox("Q2 Marks:", [2, 5, 10, 15], index=1, key="pq2_marks")
        paper_q2_ans = st.text_area("Q2 Student Answer:", value="CPU generates logical address consisting of page number and offset. Page table translates page number to frame number.", height=65, key="pq2_ans")

        col_q3_a, col_q3_b = st.columns([3, 1])
        with col_q3_a:
            paper_q3 = st.text_input("Question 3:", value=st.session_state.get("paper_q3_text", "Discuss Banker's Algorithm safety state verification."), key="pq3_text")
        with col_q3_b:
            paper_q3_m = st.selectbox("Q3 Marks:", [2, 5, 10, 15], index=2, key="pq3_marks")
        paper_q3_ans = st.text_area("Q3 Student Answer:", value="", placeholder="Leave blank to test an unanswered question...", height=65, key="pq3_ans")

        if st.button("Evaluate Complete Paper", type="primary", use_container_width=True):
            with st.spinner("Evaluating all questions..."):
                paper_data = [
                    {"question": paper_q1, "target_marks": paper_q1_m, "student_answer": paper_q1_ans},
                    {"question": paper_q2, "target_marks": paper_q2_m, "student_answer": paper_q2_ans},
                    {"question": paper_q3, "target_marks": paper_q3_m, "student_answer": paper_q3_ans}
                ]
                evaluator = StudentAnswerEvaluator(
                    api_key=active_api_key,
                    model_name=selected_model
                )
                paper_result = evaluator.evaluate_paper(paper_data)
                st.session_state.current_paper_evaluation = paper_result

        # Display Paper Evaluation Results
        if st.session_state.get("current_paper_evaluation"):
            pres = st.session_state.current_paper_evaluation
            st.divider()

            st.markdown(f"""
            <div class="sg-scorecard">
                <div class="sg-score-title">Total Score: {pres['total_marks_obtained']} / {pres['total_maximum_marks']} Marks ({pres['percentage']}%)</div>
                <div class="sg-score-meta">
                    Overall Grade: <strong>{pres['grade']}</strong> &nbsp;|&nbsp; Status: <strong>{pres['classification']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Examination Summary")
            summary_table_rows = []
            for eq in pres["question_evaluations"]:
                summary_table_rows.append({
                    "Question #": f"Q{eq['question_number']}",
                    "Question": eq["question"][:55] + ("..." if len(eq["question"]) > 55 else ""),
                    "Max Marks": f"{eq['target_marks']} M",
                    "Expected Points": f"{eq['expected_points']} pts",
                    "Awarded": f"{eq['marks_awarded']} M",
                    "Percentage": f"{eq['percentage']}%",
                    "Status": eq["status"]
                })
            st.table(summary_table_rows)

            with st.expander("Detailed Question-Wise Breakdown", expanded=False):
                for eq in pres["question_evaluations"]:
                    st.markdown(eq["report_markdown"])
                    st.divider()

            st.download_button(
                label="Download Examination Grade Sheet (Markdown)",
                data=pres["summary_report_markdown"],
                file_name=f"Examination_Grade_Sheet_{pres['total_marks_obtained']}_of_{pres['total_maximum_marks']}M.md",
                mime="text/markdown",
                key="dl_paper_eval_md"
            )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================================
# Document Explorer & Chunk Search
# =========================================================================
elif active_section == "Document Explorer":
    st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sg-section-header">
        <div class="sg-icon-box sg-icon-blue">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
        </div>
        <div>
            <div class="sg-section-title">Document Explorer & RAG Knowledge Base</div>
            <div class="sg-section-subtitle">Inspect extracted pages, raw text, and indexed passages used by the selected retrieval method</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_metadata:
        st.info("No documents indexed yet.")
    else:
        for doc in st.session_state.doc_metadata:
            with st.expander(f"{doc['source_doc']} ({doc['total_pages']} Pages, {doc['total_chunks']} Chunks)", expanded=True):
                st.caption(f"Location: `{doc['file_path']}`")
                
                page_sel = st.selectbox(
                    f"Select Page ({doc['source_doc']}):",
                    options=[p["page_number"] for p in doc["pages"]],
                    key=f"page_sel_{doc['source_doc']}"
                )
                
                selected_page_text = next((p["text"] for p in doc["pages"] if p["page_number"] == page_sel), "")
                
                p_words = len(selected_page_text.split())
                p_chars = len(selected_page_text)
                
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Words", p_words)
                col_p2.metric("Characters", p_chars)
                
                st.text_area("Extracted Plain Text:", value=selected_page_text, height=140, disabled=True, key=f"raw_{doc['source_doc']}_{page_sel}")

        st.divider()
        st.markdown("##### Retrieval Tester")
        search_query = st.text_input("Enter search query to test similarity retrieval:")
        if search_query:
            try:
                results = st.session_state.rag_engine.search(
                    search_query, 
                    top_k=5,
                    active_doc_ids=st.session_state.get("active_doc_ids")
                )
            except RetrievalError as exc:
                st.error(str(exc))
            else:
                score_label = (
                    "Cosine similarity" if st.session_state.rag_engine.retrieval_method == "semantic"
                    else "TF-IDF cosine score"
                )
                if not results:
                    st.info("No passages met the selected similarity threshold in the active document.")
                for r in results:
                    st.markdown(f"""
                    <div class="sg-source-item">
                        <strong>Rank #{r['rank']}</strong> &nbsp;|&nbsp;
                        {score_label}: <code>{r['score']}</code> &nbsp;|&nbsp; Page {r['page']} ({r['source_doc']})<br/>
                        <div style="margin-top: 4px; color: #1e293b;">{r['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# History & Records
# =========================================================================
elif active_section == "History":
    st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sg-section-header">
        <div class="sg-icon-box sg-icon-lavender">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
        </div>
        <div>
            <div class="sg-section-title">History & Saved Records</div>
            <div class="sg-section-subtitle">Local database log of generated answers, marking schemes, and student grading sessions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("##### Recent Generated Answers")
        recent_ans = list_recent_answers(limit=8)
        if not recent_ans:
            st.caption("No answers generated yet.")
        else:
            for item in recent_ans:
                with st.expander(f"[{item['marks']}M] {item['question'][:45]}..."):
                    st.caption(f"Date: {item['created_at']}")
                    st.markdown(item["answer_text"][:250] + "...")
    
    with col_h2:
        st.markdown("##### Recent Student Evaluations")
        recent_evs = list_recent_evaluations(limit=8)
        if not recent_evs:
            st.caption("No student evaluations recorded yet.")
        else:
            for ev_item in recent_evs:
                with st.expander(f"Score: {ev_item['marks_awarded']}/{ev_item['target_marks']} — {ev_item['question'][:35]}..."):
                    st.caption(f"Excerpt: {ev_item['student_answer'][:100]}...")
                    feedback_obj = ev_item.get("feedback", {})
                    st.caption(f"Grade: {feedback_obj.get('grade', 'N/A')}")

    st.markdown('</div>', unsafe_allow_html=True)
