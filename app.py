"""
app.py — Streamlit Dashboard for AI Meeting Insight Extractor
Redesigned: modern SaaS UI with PDF/DOC/DOCX upload support,
drag-and-drop area, animated progress, and card-based results.
"""

from __future__ import annotations

import json
import logging
import sys
import io
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from config import (
    ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB,
    CONVERT_TO_PDF_EXTS, AUDIO_EXTENSIONS, ML_BACKEND_URL
)
from src.gemini_layer import refine_insights

logger = logging.getLogger(__name__)

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="MeetingMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Utility: DOC/DOCX → text extraction ──────────────────
def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract raw text from a .docx file using python-docx."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        st.warning("⚠️ python-docx not installed. Run: pip install python-docx")
        return ""
    except Exception as exc:
        st.error(f"❌ Could not read DOCX: {exc}")
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file using pdfplumber (preferred) or PyPDF2."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except ImportError:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        st.warning("⚠️ No PDF library found. Run: pip install pdfplumber")
        return ""
    except Exception as exc:
        st.error(f"❌ Could not read PDF: {exc}")
        return ""


def get_file_icon(ext: str) -> str:
    return {"pdf": "📄", "doc": "📝", "docx": "📝", "txt": "📃"}.get(ext.lstrip("."), "📁")


# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:        #0d0f14;
    --surface:   #161a22;
    --border:    #252c3a;
    --accent:    #4f8ef7;
    --accent2:   #7c5ef7;
    --success:   #22d3a5;
    --warn:      #f7a84f;
    --danger:    #f75e5e;
    --text:      #e8ecf4;
    --muted:     #8896b0;
    --card-bg:   #1a1f2c;
    --radius:    14px;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
  }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  section[data-testid="stSidebar"] { display: none !important; }

  /* ── Top nav ── */
  .top-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2.5rem; border-bottom: 1px solid var(--border);
    background: rgba(13,15,20,0.95); backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }
  .nav-logo { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.25rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .nav-badge { background: var(--border); color: var(--muted); font-size: 0.72rem;
    padding: 0.2rem 0.7rem; border-radius: 20px; border: 1px solid var(--border); }

  /* ── Hero ── */
  .hero {
    text-align: center; padding: 4rem 2rem 3rem;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(79,142,247,0.07) 0%, transparent 70%);
  }
  .hero-eyebrow { color: var(--accent); font-size: 0.8rem; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 1rem; }
  .hero h1 {
    font-family: 'Syne', sans-serif; font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800; line-height: 1.15; margin: 0 0 1.2rem;
    background: linear-gradient(135deg, #e8ecf4 0%, #a0adc4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: var(--muted); font-size: 1.05rem; max-width: 540px; margin: 0 auto 2.5rem; line-height: 1.7; }

  /* ── Steps row ── */
  .steps-row { display: flex; justify-content: center; gap: 0; flex-wrap: wrap; margin-bottom: 3rem; padding: 0 2rem; }
  .step-item {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--card-bg); border: 1px solid var(--border);
    padding: 0.65rem 1.1rem; border-radius: 0; font-size: 0.85rem; color: var(--muted);
    transition: all 0.2s;
  }
  .step-item:first-child { border-radius: var(--radius) 0 0 var(--radius); }
  .step-item:last-child  { border-radius: 0 var(--radius) var(--radius) 0; }
  .step-item + .step-item { border-left: none; }
  .step-icon { font-size: 1rem; }
  .step-arrow { color: var(--border); margin: 0 -1px; z-index:1; font-size: 0.9rem; display:none; }
  .step-item:hover { background: var(--border); color: var(--text); }

  /* ── Main upload section ── */
  .upload-section { max-width: 680px; margin: 0 auto; padding: 0 2rem 4rem; }

  /* ── Drop zone ── */
  .drop-zone {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 3rem 2rem; text-align: center; cursor: pointer;
    background: var(--card-bg); transition: all 0.25s;
    position: relative; overflow: hidden;
  }
  .drop-zone::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at center, rgba(79,142,247,0.04) 0%, transparent 70%);
    pointer-events: none;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--accent); background: rgba(79,142,247,0.05);
  }
  .drop-zone .dz-icon { font-size: 2.8rem; margin-bottom: 1rem; display: block; }
  .drop-zone h3 { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; margin: 0 0 0.4rem; color: var(--text); }
  .drop-zone p  { color: var(--muted); font-size: 0.88rem; margin: 0 0 1.4rem; }
  .format-pills { display: flex; justify-content: center; gap: 0.4rem; flex-wrap: wrap; }
  .pill {
    background: var(--border); color: var(--muted); border: 1px solid rgba(255,255,255,0.06);
    padding: 0.2rem 0.65rem; border-radius: 20px; font-size: 0.75rem; font-weight: 500;
  }
  .pill.pdf { color: #f75e5e; } .pill.doc { color: var(--accent); } .pill.docx { color: var(--accent2); }

  /* ── File preview card ── */
  .file-preview {
    display: flex; align-items: center; gap: 1rem;
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1rem 1.2rem; margin-top: 1rem;
  }
  .file-icon-box {
    width: 44px; height: 44px; border-radius: 10px; display: flex;
    align-items: center; justify-content: center; font-size: 1.4rem;
    background: var(--border); flex-shrink: 0;
  }
  .file-meta { flex: 1; }
  .file-name { font-weight: 500; font-size: 0.92rem; color: var(--text); margin-bottom: 0.2rem; }
  .file-size { font-size: 0.78rem; color: var(--muted); }
  .file-status { display: flex; align-items: center; gap: 0.35rem; font-size: 0.8rem; color: var(--success); }
  .convert-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(124,94,247,0.15); color: var(--accent2);
    border: 1px solid rgba(124,94,247,0.25); border-radius: 6px;
    padding: 0.2rem 0.55rem; font-size: 0.72rem; margin-top: 0.3rem;
  }

  /* ── Progress ── */
  .progress-wrap { margin-top: 1.2rem; }
  .progress-label { display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.4rem; font-size: 0.82rem; }
  .progress-label span:first-child { color: var(--muted); }
  .progress-label span:last-child { color: var(--accent); font-weight: 600; }
  .progress-bar-track {
    height: 4px; background: var(--border); border-radius: 99px; overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    animation: shimmer 1.5s infinite;
  }
  @keyframes shimmer {
    0%   { opacity: 1; }
    50%  { opacity: 0.6; }
    100% { opacity: 1; }
  }

  /* ── Process steps indicator ── */
  .process-steps { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 1.2rem; }
  .process-step {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.65rem 1rem; border-radius: 10px; font-size: 0.85rem;
    background: var(--card-bg); border: 1px solid var(--border);
    transition: all 0.3s;
  }
  .process-step.active  { border-color: var(--accent); background: rgba(79,142,247,0.06); color: var(--text); }
  .process-step.done    { border-color: var(--success); background: rgba(34,211,165,0.05); color: var(--success); }
  .process-step.pending { color: var(--muted); }
  .step-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: var(--border);
  }
  .process-step.active .step-dot  { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 1s infinite; }
  .process-step.done .step-dot    { background: var(--success); }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

  /* ── Error / Warning / Success alerts ── */
  .alert {
    padding: 0.8rem 1rem; border-radius: 10px; font-size: 0.88rem;
    display: flex; align-items: flex-start; gap: 0.6rem; margin-top: 0.8rem;
  }
  .alert-error   { background: rgba(247,94,94,0.1);  border: 1px solid rgba(247,94,94,0.25);  color: #f75e5e; }
  .alert-warning { background: rgba(247,168,79,0.1); border: 1px solid rgba(247,168,79,0.25); color: var(--warn); }
  .alert-success { background: rgba(34,211,165,0.1); border: 1px solid rgba(34,211,165,0.25); color: var(--success); }

  /* ── Analyse button ── */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border: none !important; border-radius: 10px !important; padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important;
    color: white !important; width: 100% !important; margin-top: 1.2rem !important;
    box-shadow: 0 4px 20px rgba(79,142,247,0.25) !important;
    transition: all 0.2s !important; cursor: pointer !important;
  }
  .stButton > button[kind="primary"]:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 28px rgba(79,142,247,0.35) !important; }

  /* ── Mode toggle ── */
  .stRadio > div { flex-direction: row !important; gap: 0.5rem !important; }
  .stRadio label { background: var(--card-bg) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; padding: 0.4rem 0.9rem !important; cursor: pointer !important;
    font-size: 0.85rem !important; color: var(--muted) !important; transition: all 0.2s !important; }
  .stRadio label:has(input:checked) { border-color: var(--accent) !important; color: var(--accent) !important; background: rgba(79,142,247,0.07) !important; }

  /* ── Textarea ── */
  .stTextArea textarea {
    background: var(--card-bg) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
  }
  .stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: var(--card-bg) !important; border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important; padding: 0.5rem !important;
  }
  [data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
  [data-testid="stFileUploadDropzone"] { background: transparent !important; }

  /* ── Results section ── */
  .results-header {
    padding: 2rem 2.5rem 1rem;
    border-top: 1px solid var(--border);
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(124,94,247,0.05) 0%, transparent 70%);
  }
  .results-header h2 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.6rem; margin: 0 0 0.3rem; }
  .results-header p  { color: var(--muted); font-size: 0.88rem; margin: 0; }

  /* ── Summary card ── */
  .summary-card {
    background: linear-gradient(135deg, rgba(79,142,247,0.1) 0%, rgba(124,94,247,0.08) 100%);
    border: 1px solid rgba(79,142,247,0.2); border-radius: var(--radius);
    padding: 1.4rem 1.6rem; margin-bottom: 1.5rem;
  }
  .summary-card h2 { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: var(--text); margin: 0 0 0.8rem; }
  .summary-meta { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 1rem; }
  .meta-item { font-size: 0.85rem; color: var(--muted); }
  .meta-item strong { color: var(--text); display: block; font-size: 0.95rem; margin-bottom: 0.1rem; }
  .outcome-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-top: 1rem; }
  .outcome-item {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.7rem 1rem; text-align: center;
  }
  .outcome-num { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; color: var(--accent); line-height: 1; }
  .outcome-label { font-size: 0.72rem; color: var(--muted); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }

  /* ── Insight card ── */
  .insight-card {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.2rem 1.4rem; margin-bottom: 1rem; transition: all 0.2s;
  }
  .insight-card:hover { border-color: rgba(79,142,247,0.3); box-shadow: 0 4px 24px rgba(79,142,247,0.07); }
  .insight-card-header {
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.9rem;
    padding-bottom: 0.7rem; border-bottom: 1px solid var(--border);
  }
  .insight-card-header .card-icon { font-size: 1rem; }
  .insight-card-header h3 { font-family: 'Syne', sans-serif; font-size: 0.92rem; font-weight: 700;
    color: var(--text) !important; margin: 0; text-transform: uppercase; letter-spacing: 0.06em; }
  .insight-card ul { margin: 0; padding: 0; list-style: none; }
  .insight-card li {
    padding: 0.35rem 0 0.35rem 1rem; font-size: 0.88rem; color: #c8d0e0 !important;
    border-bottom: 1px solid var(--border); position: relative; line-height: 1.55;
  }
  .insight-card li:last-child { border-bottom: none; }
  .insight-card li::before { content: "›"; position: absolute; left: 0; color: var(--accent); font-weight: 700; }
  .insight-card em { color: var(--muted) !important; font-style: italic; }

  /* ── Score card ── */
  .score-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 42px; height: 42px; border-radius: 50%; font-family: 'Syne',sans-serif;
    font-size: 0.9rem; font-weight: 800; color: white; flex-shrink: 0;
  }
  .score-high   { background: linear-gradient(135deg, #22d3a5, #059669); }
  .score-medium { background: linear-gradient(135deg, #f7a84f, #d97706); }
  .score-low    { background: linear-gradient(135deg, #f75e5e, #dc2626); }

  /* ── Resp rows ── */
  .resp-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px solid var(--border);
    font-size: 0.87rem; color: #c8d0e0 !important;
  }
  .resp-row:last-child { border-bottom: none; }
  .resp-count {
    background: rgba(79,142,247,0.15); color: var(--accent) !important;
    border: 1px solid rgba(79,142,247,0.2); border-radius: 20px;
    padding: 0.1rem 0.65rem; font-size: 0.78rem; font-weight: 600;
  }

  /* ── Export button ── */
  .stDownloadButton > button {
    background: var(--card-bg) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
    font-weight: 500 !important; width: 100% !important; transition: all 0.2s !important;
  }
  .stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid var(--border) !important; margin: 2rem 0 !important; }

  /* ── Spinner text ── */
  .stSpinner > div { border-top-color: var(--accent) !important; }

  /* ── Mobile ── */
  @media (max-width: 640px) {
    .outcome-grid { grid-template-columns: repeat(2, 1fr); }
    .top-nav { padding: 0.8rem 1.2rem; }
    .hero { padding: 2.5rem 1.2rem 2rem; }
    .upload-section { padding: 0 1.2rem 3rem; }
    .results-header { padding: 1.5rem 1.2rem 0.8rem; }
    .steps-row { gap: 0.3rem; }
    .step-item { padding: 0.5rem 0.7rem; font-size: 0.78rem; }
  }
</style>
""", unsafe_allow_html=True)

# ── Top Nav ───────────────────────────────────────────────
st.markdown("""
<nav class="top-nav">
  <span class="nav-logo">🧠 MeetingMind</span>
  <span class="nav-badge">AI · v2.0</span>
</nav>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<section class="hero">
  <p class="hero-eyebrow">AI-Powered Meeting Intelligence</p>
  <h1>Transform Meetings Into<br>Actionable Insights</h1>
  <p>Upload your meeting transcript or document. Our AI extracts decisions, action items,
     deadlines, and key insights — automatically.</p>
</section>
""", unsafe_allow_html=True)

# ── Steps row ─────────────────────────────────────────────
st.markdown("""
<div class="steps-row">
  <div class="step-item"><span class="step-icon">📤</span> Upload Document</div>
  <div class="step-item"><span class="step-icon">⚙️</span> Process &amp; OCR</div>
  <div class="step-item"><span class="step-icon">✨</span> Summarise</div>
  <div class="step-item"><span class="step-icon">💬</span> Extract Insights</div>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

# Mode toggle
col_a, col_b = st.columns([1.2, 0.8])
with col_a:
    input_mode = st.radio("Input method", ["📄 File", "🎙️ Audio", "📝 Text"], horizontal=True, label_visibility="collapsed")

st.markdown("---")

transcript_text = ""
uploaded_name   = ""

if input_mode == "📄 File":
    # Drop zone visual header
    st.markdown(f"""
    <div style="border:2px dashed #252c3a;border-radius:14px;padding:2rem 1.5rem 1.5rem;background:#161a22;text-align:center;margin-bottom:0.75rem;">
      <span style="font-size:2.5rem;display:block;margin-bottom:0.8rem;">📂</span>
      <h3 style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8ecf4;margin:0 0 0.3rem;">
        Drag &amp; drop your document here
      </h3>
      <p style="color:#8896b0;font-size:0.83rem;margin:0 0 1rem;">or click to browse files</p>
      <div class="format-pills" style="display:flex;justify-content:center;gap:0.4rem;">
        <span class="pill pdf">PDF</span>
        <span class="pill doc">DOC</span>
        <span class="pill docx">DOCX</span>
        <span class="pill" style="color:#8896b0;">TXT</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload transcript",
        type=["pdf", "doc", "docx", "txt"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        ext      = Path(uploaded.name).suffix.lower()
        size_mb  = uploaded.size / (1024 * 1024)
        icon     = get_file_icon(ext)
        needs_cv = ext in CONVERT_TO_PDF_EXTS

        if size_mb > MAX_UPLOAD_SIZE_MB:
            st.markdown(f'<div class="alert alert-error">⚠️ File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_SIZE_MB} MB.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="file-preview"><div class="file-icon-box">{icon}</div><div class="file-meta"><div class="file-name">{uploaded.name}</div><div class="file-size">{size_mb:.2f} MB · {ext.upper()}</div></div><div class="file-status">✓ Ready</div></div>', unsafe_allow_html=True)
            uploaded_name = uploaded.name
            file_bytes = uploaded.read()

            if ext == ".txt":
                try: transcript_text = file_bytes.decode("utf-8")
                except: st.markdown('<div class="alert alert-error">❌ Encoding error.</div>', unsafe_allow_html=True)
            elif ext == ".pdf":
                with st.spinner("📄 Extracting PDF text..."): transcript_text = extract_text_from_pdf(file_bytes)
            elif ext in (".doc", ".docx"):
                with st.spinner("🔄 Extracting DOCX text..."): transcript_text = extract_text_from_docx(file_bytes)

elif input_mode == "🎙️ Audio":
    st.markdown(f"""
    <div style="border:2px dashed #252c3a;border-radius:14px;padding:2rem 1.5rem 1.5rem;background:#161a22;text-align:center;margin-bottom:0.75rem;">
      <span style="font-size:2.5rem;display:block;margin-bottom:0.8rem;">🎙️</span>
      <h3 style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8ecf4;margin:0 0 0.3rem;">
        Upload Audio Recording
      </h3>
      <p style="color:#8896b0;font-size:0.83rem;margin:0 0 1rem;">Processed via Whisper ML Backend</p>
      <div class="format-pills" style="display:flex;justify-content:center;gap:0.4rem;">
        <span class="pill" style="color:var(--accent); border-color:var(--accent);">WAV</span>
        <span class="pill" style="color:var(--accent); border-color:var(--accent);">MP3</span>
        <span class="pill" style="color:var(--accent); border-color:var(--accent);">M4A</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_audio = st.file_uploader(
        "Upload audio",
        type=[ext.strip(".") for ext in AUDIO_EXTENSIONS],
        label_visibility="collapsed"
    )

    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("✨ Transcribe Audio"):
            with st.spinner("🧠 Whisper is transcribing... (this may take a minute)"):
                try:
                    files = {"file": (uploaded_audio.name, uploaded_audio, uploaded_audio.type)}
                    resp = requests.post(f"{ML_BACKEND_URL}/api/v1/transcribe", files=files, timeout=300)
                    if resp.status_code == 200:
                        transcript_text = resp.json().get("text", "")
                        st.markdown(f'<div class="alert alert-success">✅ Transcription complete! {len(transcript_text)} chars.</div>', unsafe_allow_html=True)
                        st.session_state["audio_transcript"] = transcript_text
                    else:
                        st.error(f"Backend Error: {resp.text}")
                except Exception as e:
                    st.error(f"Could not connect to ML Backend: {e}")

    if "audio_transcript" in st.session_state:
        transcript_text = st.session_state["audio_transcript"]
        with st.expander("📝 View Transcription"):
            st.write(transcript_text)

else:
    transcript_text = st.text_area(
        "Paste your meeting transcript",
        height=240,
        placeholder="John: Let's get started. The main agenda today is reviewing Q3 targets...\nSarah: We agreed to move the launch date to November 15th...",
        label_visibility="collapsed"
    )

# ── Analyse Button ────────────────────────────────────────
analyse_clicked = st.button("🚀 Analyse Meeting", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)  # close upload-section

# ── Analysis Pipeline ─────────────────────────────────────
if analyse_clicked:
    if not transcript_text.strip():
        st.markdown('<div style="max-width:680px;margin:0 auto;padding:0 2rem;"><div class="alert alert-warning">⚠️ Please upload a document or paste transcript text before analysing.</div></div>', unsafe_allow_html=True)
    else:
        # Animated progress steps
        with st.spinner(""):
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style="max-width:680px;margin:0 auto;padding:0 2rem;">
              <div class="process-steps">
                <div class="process-step done">  <div class="step-dot"></div>📤 Document loaded</div>
                <div class="process-step active"><div class="step-dot"></div>⚙️ Running NLP pipeline…</div>
                <div class="process-step pending"><div class="step-dot"></div>✨ Extracting insights</div>
                <div class="process-step pending"><div class="step-dot"></div>📊 Building report</div>
              </div>
              <div class="progress-wrap">
                <div class="progress-label"><span>Processing…</span><span>analyzing</span></div>
                <div class="progress-bar-track"><div class="progress-bar-fill" style="width:55%"></div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                from src.insight_extractor import extract_insights, insights_to_json
                insights = extract_insights(transcript_text)
            except Exception as exc:
                progress_placeholder.empty()
                logger.error("Pipeline error: %s", exc, exc_info=True)
                st.markdown(f'<div style="max-width:680px;margin:0 auto;padding:0 2rem;"><div class="alert alert-error">❌ Analysis failed: <code>{exc}</code><br>Check dependencies and model training.</div></div>', unsafe_allow_html=True)
                st.stop()

        progress_placeholder.empty()

        if insights.get("errors"):
            for err in insights["errors"]:
                st.markdown(f'<div style="max-width:1100px;margin:0 auto;padding:0 2.5rem;"><div class="alert alert-warning">⚠️ {err}</div></div>', unsafe_allow_html=True)

        # ── Results Header ──────────────────────────────
        st.markdown("""
        <div class="results-header">
          <h2>📋 Meeting Insights</h2>
          <p>AI-extracted decisions, actions, and intelligence from your transcript.</p>
        </div>
        """, unsafe_allow_html=True)

        # Pad the results in a container
        st.markdown('<div style="padding:1.5rem 2.5rem 4rem;max-width:1200px;">', unsafe_allow_html=True)

        # ── Summary Card ────────────────────────────────
        summary      = insights.get("meeting_summary", {})
        title        = summary.get("title", "General Meeting")
        participants = summary.get("participants", [])
        parts_str    = ", ".join(participants) if participants else "Not detected"
        word_count   = len(transcript_text.split())
        duration     = max(1, round(word_count / 150))
        n_dec = len(insights.get("decisions", []))
        n_act = len(insights.get("action_items", []))
        n_dl  = len(insights.get("deadlines", []))
        n_iss = len(insights.get("open_issues", []))

        st.markdown(f"""
        <div class="summary-card">
          <h2>📌 {title}</h2>
          <div class="summary-meta">
            <div class="meta-item"><strong>~{duration} mins</strong>Estimated duration</div>
            <div class="meta-item"><strong>{word_count:,} words</strong>Transcript length</div>
            <div class="meta-item"><strong>{parts_str}</strong>Participants</div>
          </div>
          <div class="outcome-grid">
            <div class="outcome-item"><div class="outcome-num">{n_dec}</div><div class="outcome-label">Decisions</div></div>
            <div class="outcome-item"><div class="outcome-num">{n_act}</div><div class="outcome-label">Action Items</div></div>
            <div class="outcome-item"><div class="outcome-num">{n_dl}</div><div class="outcome-label">Deadlines</div></div>
            <div class="outcome-item"><div class="outcome-num">{n_iss}</div><div class="outcome-label">Open Issues</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column detail grid ───────────────────────
        left_col, right_col = st.columns(2, gap="medium")

        with left_col:
            # Deadlines
            deadlines = insights.get("deadlines", [])
            items = "".join(f'<li>{dl.get("description","")}</li>' for dl in deadlines) if deadlines else "<li><em>None detected</em></li>"
            st.markdown(f'<div class="insight-card"><div class="insight-card-header"><span class="card-icon">⏰</span><h3>Deadlines</h3></div><ul>{items}</ul></div>', unsafe_allow_html=True)

            # Responsibility map
            resp = insights.get("responsibility_map", {})
            if resp:
                rows = "".join(
                    f'<div class="resp-row"><span>{p}</span><span class="resp-count">{c} task{"s" if c!=1 else ""}</span></div>'
                    for p, c in sorted(resp.items())
                )
                st.markdown(f'<div class="insight-card"><div class="insight-card-header"><span class="card-icon">👥</span><h3>Responsibility Map</h3></div>{rows}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-card"><div class="insight-card-header"><span class="card-icon">👥</span><h3>Responsibility Map</h3></div><p style="color:var(--muted);font-size:0.87rem;"><em>No task assignments detected.</em></p></div>', unsafe_allow_html=True)

            # Intelligence score
            intel    = insights.get("intelligence_score", {})
            score    = intel.get("score", 0)
            flags    = intel.get("flags", [])
            s_cls    = "score-high" if score >= 70 else "score-medium" if score >= 40 else "score-low"
            flags_html = "".join(f'<li>{i} {t}</li>' for i, t in flags) if flags else "<li><em>No flags</em></li>"
            st.markdown(f"""
            <div class="insight-card">
              <div class="insight-card-header">
                <span class="card-icon">📈</span>
                <h3>Intelligence Score</h3>
                <span class="score-badge {s_cls}" style="margin-left:auto">{score}</span>
              </div>
              <ul>{flags_html}</ul>
            </div>""", unsafe_allow_html=True)

        with right_col:
            # Decisions
            decisions = insights.get("decisions", [])
            items = "".join(f"<li>{d}</li>" for d in decisions) if decisions else "<li><em>None detected</em></li>"
            st.markdown(f'<div class="insight-card"><div class="insight-card-header"><span class="card-icon">✅</span><h3>Decisions Made</h3></div><ul>{items}</ul></div>', unsafe_allow_html=True)

            # Action items
            actions = insights.get("action_items", [])
            if actions:
                items = ""
                for a in actions:
                    assignee = a.get("assignee", "—")
                    task     = a.get("task", "")
                    dl_tag   = f" <span style='color:var(--warn);font-size:0.78rem'>(by {a['deadline']})</span>" if a.get("deadline") else ""
                    items   += f"<li><strong style='color:var(--accent)'>{assignee}</strong> → {task}{dl_tag}</li>"
            else:
                items = "<li><em>None detected</em></li>"
            st.markdown(f'<div class="insight-card"><div class="insight-card-header"><span class="card-icon">📌</span><h3>Action Items</h3></div><ul>{items}</ul></div>', unsafe_allow_html=True)

            # Open issues
            issues = insights.get("open_issues", [])
            items  = "".join(f"<li>{i}</li>" for i in issues) if issues else "<li><em>None detected</em></li>"
            st.markdown(f'<div class="insight-card"><div class="insight-card-header"><span class="card-icon">⚠️</span><h3>Open Issues</h3></div><ul>{items}</ul></div>', unsafe_allow_html=True)

        # ── Gemini Reframing Layer ───────────────────────
        st.markdown("### ✨ Gemini AI Insights")
        if st.button("🪄 Refine Analysis with Gemini"):
            with st.spinner("Gemini is polishing your report..."):
                refined_report = refine_insights(insights)
                st.session_state["gemini_report"] = refined_report

        if "gemini_report" in st.session_state:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2333 0%, #161a22 100%); border: 1px solid #2a3448; border-radius: 16px; padding: 2rem; margin-top: 1rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
              <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.75rem;">✨</span>
                <h3 style="margin: 0; color: #fff; font-family: 'Syne', sans-serif;">Gemini Refined Report</h3>
                <span style="background: #a855f7; color: #fff; font-size: 0.65rem; padding: 2px 8px; border-radius: 20px; font-weight: 700; text-transform: uppercase; margin-left: 1rem;">Premium</span>
              </div>
              <div style="color: #cbd5e1; line-height: 1.6; font-size: 0.95rem;">
                {st.session_state["gemini_report"]}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Export ───────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        json_str = insights_to_json(insights)
        dl_col, _ = st.columns([1, 2])
        with dl_col:
            st.download_button("📥 Download Raw Insights (JSON)", data=json_str,
                               file_name="meeting_insights.json", mime="application/json",
                               use_container_width=True)
        with st.expander("🔍 View Raw Analysis JSON"):
            st.json(json.loads(json_str))

        st.markdown('</div>', unsafe_allow_html=True)
