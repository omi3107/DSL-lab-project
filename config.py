"""
config.py — Central configuration for AI Meeting Insight Extractor.

All paths, model names, label sets, and tunables live here so that
every other module imports from a single source of truth.
"""

from pathlib import Path
import logging

# ── Project Root ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data Directories ─────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
UPLOAD_DIR      = DATA_DIR / "uploads"

# Auto-create directories if missing
for _dir in (RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, UPLOAD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Label Set (Sentence Classification) ──────────────────
LABELS = [
    "DECISION",
    "ACTION_ITEM",
    "DEADLINE",
    "DISCUSSION",
    "OTHER",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# ── spaCy ────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Transformer Model ───────────────────────────────────
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_EPOCHS     = 5
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_LR         = 2e-5

# ── TF-IDF Classifier ───────────────────────────────────
TFIDF_MAX_FEATURES = 5000
CV_FOLDS           = 5          # Cross-validation folds

# ── NLP Analysis ─────────────────────────────────────────
TOP_N_KEYWORDS     = 10
NER_ENTITY_TYPES   = ["PERSON", "DATE", "ORG", "TIME", "GPE"]

# ── File Upload ──────────────────────────────────────────
MAX_UPLOAD_SIZE_MB  = 10
# Supported upload formats: PDF, DOC, DOCX, TXT
# DOC/DOCX are auto-converted to PDF before processing
ALLOWED_EXTENSIONS  = [".pdf", ".doc", ".docx", ".txt"]
ALLOWED_MIME_TYPES  = {
    ".pdf":  "application/pdf",
    ".doc":  "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt":  "text/plain",
}
# Extensions that require conversion to PDF before OCR/text extraction
CONVERT_TO_PDF_EXTS = [".doc", ".docx"]

# ── Audio Upload ─────────────────────────────────────────
AUDIO_EXTENSIONS    = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
ML_BACKEND_URL      = "http://localhost:8000"

# ── Logging ──────────────────────────────────────────────
LOG_LEVEL  = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("meeting_insights")
