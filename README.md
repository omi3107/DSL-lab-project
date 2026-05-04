# MeetingMind — System Overview

> **AI-powered meeting transcript analyser** that combines a custom-trained ML classification pipeline with LLM-based intelligence to extract actionable insights from meeting transcripts.

---

## Table of Contents

1. [Application Summary](#1-application-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Core Pipeline](#5-core-pipeline)
6. [Frontend Dashboard](#6-frontend-dashboard)
7. [Backend API](#7-backend-api)
8. [ML Classification Pipeline](#8-ml-classification-pipeline)
9. [AI Framing Layer](#9-ai-framing-layer)
10. [Data Flow](#10-data-flow)
11. [External Integrations](#11-external-integrations)
12. [Configuration & Environment](#12-configuration--environment)
13. [How to Run](#13-how-to-run)

---

## 1. Application Summary

**MeetingMind** is a full-stack AI meeting analysis application. Users upload meeting transcripts (PDF, DOCX, TXT, or raw text), and the system:

1. **Classifies** each sentence using a trained SVM model into five categories: *Decision*, *Task*, *Deadline*, *Issue*, or *General Discussion*.
2. **Refines** the ML output using an LLM (Gemini or Groq) to produce clean, professional insights — including participant detection, responsibility mapping, meeting title generation, and an intelligence score.
3. **Displays** everything in a premium, dark-themed dashboard with interactive visualizations, history tracking, AI summaries, Notion export, and multilingual support.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Browser)                          │
│  meetingmind.html — Single-page app served via Streamlit        │
│  ┌───────────┬────────────┬─────────────┬──────────────────┐    │
│  │   Home    │  History   │ AI Summary  │    Settings      │    │
│  │ (Upload & │ (Past      │ (Generate   │ (Theme, Export,  │    │
│  │  Analyse) │  Meetings) │  summaries) │  Language)       │    │
│  └───────────┴────────────┴─────────────┴──────────────────┘    │
│  localStorage: meetingmind_history, meetingmind_settings        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP (fetch)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI, port 8502)                  │
│                                                                 │
│  POST /api/analyse       ← Full ML + AI pipeline               │
│  POST /api/extract_text  ← PDF/DOCX/TXT → plain text           │
│  POST /api/export_notion ← Export insights to Notion            │
│  GET  /api/notion_status ← Check Notion connection              │
│  GET  /api/health        ← Health check                         │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  ML Pipeline     │───▶│  AI Framing Layer                │   │
│  │  (SVM + TF-IDF)  │    │  (Gemini → Groq → Fallback)     │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXTERNAL SERVICES                             │
│  • Google Gemini API (gemini-2.0-flash) — Primary LLM           │
│  • Groq API (llama-3.3-70b-versatile) — Fallback LLM           │
│  • Notion API — Meeting export                                  │
│  • Google Translate Widget — UI translation                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Layer         | Technology                                                    |
|---------------|---------------------------------------------------------------|
| **Frontend**  | Vanilla HTML/CSS/JS (single-file), embedded in Streamlit      |
| **Hosting**   | Streamlit (`app.py`) — wraps HTML as a full-screen component  |
| **Backend**   | FastAPI + Uvicorn (Python)                                    |
| **ML Model**  | scikit-learn (TF-IDF + Linear SVM with CalibratedClassifierCV)|
| **NLP**       | NLTK (tokenization, lemmatization, stopwords)                 |
| **AI/LLM**    | Google Gemini (`google-genai`), Groq (`groq`)                 |
| **File I/O**  | pdfplumber (PDF), python-docx (DOCX)                          |
| **Export**     | Notion API via httpx                                          |
| **Fonts**     | Google Fonts: Syne (headings), DM Sans (body)                 |
| **Theming**   | CSS custom properties with light/dark mode support            |

---

## 4. Project Structure

```
MeetingMind-AI-Redesigned/
│
├── app.py                          # Streamlit entry point — serves the HTML
├── config.py                       # Central config (labels, paths, model params)
├── meetingmind.html                # Complete frontend: HTML + CSS + JS (~4050 lines)
├── requirements.txt                # Python dependencies (root)
├── .env                            # API keys (Gemini, Groq, Notion) — gitignored
├── .gitignore                      # Version control exclusions
│
├── src/                            # Source modules
│   ├── insight_extractor.py        #   ML inference wrapper — sentence classification
│   └── gemini_layer.py             #   AI framing layer — Gemini/Groq refinement
│
├── backend/
│   ├── api/
│   │   ├── analysis_server.py      #   FastAPI server — all REST endpoints
│   │   └── notion_export.py        #   Notion page builder & API client
│   │
│   ├── ml_model/                   #   Complete ML pipeline
│   │   ├── dataset/
│   │   │   ├── labelled_data.csv   #     Training dataset (~47 MB)
│   │   │   ├── data_generator.py   #     Synthetic data generation
│   │   │   ├── generate_dataset.py #     Dataset creation script
│   │   │   └── balance_dataset.py  #     Class balancing utilities
│   │   │
│   │   ├── preprocessing/
│   │   │   └── text_cleaner.py     #     Text cleaning (lowercase, lemma, stopwords)
│   │   │
│   │   ├── models/
│   │   │   ├── tfidf_vectorizer.py #     TF-IDF vectorizer (unigrams + bigrams)
│   │   │   ├── baseline_classifier.py #  Linear SVM classifier
│   │   │   ├── model_utils.py      #     Save/load artifacts (joblib)
│   │   │   └── saved/              #     Serialized model files (.joblib)
│   │   │
│   │   ├── training/
│   │   │   ├── train_baseline.py   #     Training script with EDA & visualization
│   │   │   ├── evaluate_baseline.py#     Evaluation metrics (accuracy, F1, confusion)
│   │   │   ├── save_model.py       #     Model persistence script
│   │   │   └── results/            #     Training output & plots
│   │   │
│   │   └── inference/
│   │       └── predict_text.py     #     SentenceClassifier class (batch & single)
│   │
│   └── dl_model/                   #   Deep learning model (Whisper fine-tune — reserved)
│       ├── audio_data/
│       ├── checkpoints/
│       ├── inference/
│       ├── training/
│       └── whisper_finetune/
│
├── transcripts/                    # Sample meeting transcripts (PDF, DOCX)
├── github_raw_data/                # Raw research datasets (AMI, Google, MeetingBank)
└── history.json                    # Sample history data for development
```

---

## 5. Core Pipeline

The analysis pipeline follows a three-stage architecture:

### Stage 1 — Text Extraction
- **File upload**: Frontend sends file to `POST /api/extract_text`
- **Text input**: Raw text sent directly
- Supports: PDF (pdfplumber), DOCX (python-docx), TXT (utf-8 decode)

### Stage 2 — ML Classification (`insight_extractor.py`)
- Splits transcript into sentences using NLTK `sent_tokenize`
- Each sentence is cleaned: lowercased → URLs removed → punctuation stripped → stopwords removed → lemmatized
- Sentences are vectorized using a pre-trained **TF-IDF vectorizer** (15K features, unigram + bigram)
- Classified by a **Calibrated Linear SVM** into one of 5 labels:
  - `Decision` — Key decisions made during the meeting
  - `Task` — Action items assigned to participants
  - `Deadline` — Time-bound commitments
  - `Issue` — Blockers, risks, or concerns
  - `General` — General discussion points
- Calculates an **intelligence score** (0–100) based on the volume and diversity of classified insights
- Returns structured dict with grouped sentences + metadata

### Stage 3 — AI Framing (`gemini_layer.py`)
- Takes the raw ML output and a detailed JSON-structured prompt
- Sends to **Groq** (primary) → **Gemini** (fallback) → **basic formatting** (last resort)
- The LLM refines and returns:
  - Clean, professional bullet points per category
  - **Participant names** extracted from transcript context
  - **Responsibility map** (participant → assigned tasks)
  - **Meeting title** (auto-generated)
  - **Intelligence score** with assessment
- Supports **multilingual output** — all text is generated in the user's selected language

---

## 6. Frontend Dashboard

The frontend is a single monolithic HTML file (`meetingmind.html`, ~4050 lines) containing all CSS, HTML structure, and JavaScript logic inline.

### Navigation (Left Sidebar)
| Icon | Section | Description |
|------|---------|-------------|
| 🏠 | **Home** | Upload & analyse transcripts |
| 🕐 | **History** | Browse, search, filter past analyses |
| ⚙️ | **Settings** | Appearance, language, export preferences |
| ⚡ | **AI Summary** | Generate formatted summaries of past meetings |

### Home View
- **File upload zone**: Drag & drop or click to upload (PDF, DOCX, TXT; max 10 MB)
- **Text input tab**: Paste raw transcript text directly
- **Analysis pipeline indicators**: Visual step pills (Classification → Entity Extraction → Report Generated)
- **Results display**: After analysis, shows:
  - Meeting title
  - Participant list
  - Categorized insights (Decisions, Tasks, Deadlines, Issues, General)
  - Responsibility map
  - Intelligence score
  - Confidence level (line graph visualization)
  - Quality tags (e.g., "Structured", "Action-Heavy", "High Risk")
- **Export buttons**: New Analysis, Export (JSON/MD), Notion export
- **Language note**: Indicates output will be generated in selected language

### History View
- Split-pane layout: meeting list (left) + detail panel (right)
- Search bar + filters (Score, Category, Time)
- Meeting cards with date, title, participants, and score badge
- Detailed view shows all categorized insights
- Delete individual entries
- Data persisted in `localStorage` (`meetingmind_history`)

### AI Summary View
- Split-pane layout matching History
- Search bar for filtering meetings
- "Generate Summary" button on each meeting card
- Right panel shows AI-generated summary with **typing animation** (Claude-style streaming effect)
- Summary is structured with colored section headers:
  - 📄 Overview (participants, score)
  - ✅ Key Decisions
  - 🎯 Action Items (with assignees and deadlines)
  - ⚠️ Blockers & Issues
  - 💬 General Discussion
- **No additional API calls** — generates from localStorage data on the client side

### Settings View
- **Appearance**: Light / Dark / System mode toggle
- **Export format**: JSON or Markdown (default for downloads)
- **Language**: Dropdown with 10 supported languages (English, Hindi, Spanish, French, German, Japanese, Chinese, Korean, Arabic, Portuguese)
- **Auto-export to Notion**: Toggle to automatically push analyses to Notion

### Design System
- **Color palette**: Custom dark theme with CSS custom properties (`--bg`, `--accent`, `--success`, etc.)
- **Light mode**: Full light theme via `.light-mode` class
- **Typography**: Syne (headings), DM Sans (body text)
- **Animations**: Smooth transitions, glassmorphism effects, micro-interactions
- **Responsive scrollbars**: Custom 3px scrollbar styling

---

## 7. Backend API

**Server**: FastAPI on port `8502`, started via Uvicorn

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyse` | Full ML + AI analysis pipeline. Accepts `{text, target_language}`. Returns structured insights. |
| `POST` | `/api/extract_text` | Extracts plain text from uploaded file (PDF/DOCX/TXT). Returns `{text}`. |
| `POST` | `/api/export_notion` | Creates a formatted Notion page from meeting data. Returns `{status, url, page_id}`. |
| `GET`  | `/api/notion_status` | Checks if Notion credentials are configured and valid. |
| `GET`  | `/api/health` | Health check — returns pipeline status. |

### Response Schema (`/api/analyse`)
```json
{
  "participants": ["Alice", "Bob"],
  "decisions": ["Decided to use React for the frontend"],
  "tasks": [{"who": "Alice", "task": "Prepare slides", "by": "Friday"}],
  "deadlines": ["Submit report by end of week"],
  "issues": ["API integration is still pending"],
  "general": ["Discussed next sprint priorities"],
  "score": 75,
  "confidence": 80,
  "tags": ["Structured", "Action-Heavy"],
  "title": "Sprint Planning Review",
  "responsibility_map": {"Alice": ["Prepare slides"]},
  "ai_provider": "Gemini"
}
```

---

## 8. ML Classification Pipeline

### Architecture

```
Raw Text → Sentence Tokenization (NLTK) → Text Cleaning → TF-IDF Vectorization → SVM Classification
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| **Text Cleaner** | `preprocessing/text_cleaner.py` | Lowercase → URL removal → punctuation removal → stopword removal → WordNet lemmatization |
| **TF-IDF Vectorizer** | `models/tfidf_vectorizer.py` | 15,000 features, unigrams + bigrams, sublinear TF, min_df=2, max_df=0.95 |
| **SVM Classifier** | `models/baseline_classifier.py` | `LinearSVC` (C=1.0, balanced class weights) wrapped in `CalibratedClassifierCV` for probability estimates |
| **Model Utils** | `models/model_utils.py` | joblib-based save/load for vectorizer, classifier, and label encoder |
| **Inference** | `inference/predict_text.py` | `SentenceClassifier` class with `predict()` and `predict_batch()` methods; also provides CLI |

### Labels
| Label | Description |
|-------|-------------|
| `Decision` | Key decisions made during the meeting |
| `Task` | Action items or assignments |
| `Deadline` | Time-bound commitments or due dates |
| `Issue` | Blockers, risks, or open concerns |
| `General` | General discussion, context, or filler |

### Training Data
- Located at `backend/ml_model/dataset/labelled_data.csv` (~47 MB)
- Includes synthetic data generation (`data_generator.py`) and class balancing (`balance_dataset.py`)
- Training script: `training/train_baseline.py` (includes EDA, visualization, cross-validation)
- Evaluation: `training/evaluate_baseline.py` (accuracy, precision, recall, F1, confusion matrix)

### Saved Artifacts
Stored in `backend/ml_model/models/saved/`:
- `tfidf_vectorizer.joblib`
- `svm_classifier.joblib`
- `label_encoder.joblib`

---

## 9. AI Framing Layer

### Purpose
Transforms raw ML-classified output into clean, professional, display-ready content using LLMs.

### Provider Priority
1. **Groq** (primary) — `llama-3.3-70b-versatile`, temperature 0.3
2. **Gemini** (fallback) — `gemini-2.0-flash`
3. **Basic formatting** (last resort) — Returns raw ML output without refinement

### Capabilities
- Refines ML bullet points into clean, concise sentences
- Extracts participant names from transcript context
- Maps responsibilities (participant → task assignments)
- Generates descriptive meeting title
- Calculates intelligence score (0–100) with assessment
- **Multilingual**: Generates all output in the user's selected language
- **AI Summary generation**: Separate function for executive summaries (also uses Groq → Gemini → fallback chain)

---

## 10. Data Flow

```
User uploads file or pastes text
        │
        ▼
Frontend sends file to POST /api/extract_text (if file)
        │
        ▼
Frontend sends text + language to POST /api/analyse
        │
        ▼
Backend: insight_extractor.extract_insights(text)
  ├─ NLTK sent_tokenize → sentence list
  ├─ text_cleaner.clean_text() → cleaned sentences
  ├─ TF-IDF vectorizer → sparse feature matrix
  ├─ SVM classifier → predicted labels + confidence
  └─ Returns structured dict with grouped insights
        │
        ▼
Backend: gemini_layer.refine_insights(raw, language)
  ├─ Builds structured prompt with ML output + transcript snippet
  ├─ Tries Groq → Gemini → Fallback
  └─ Returns refined JSON (title, participants, decisions, tasks, etc.)
        │
        ▼
Backend: _build_response() merges ML + AI results
        │
        ▼
Frontend receives JSON → renders dashboard
  ├─ Updates category panels (decisions, tasks, issues, etc.)
  ├─ Draws confidence graph
  ├─ Shows tags and intelligence score
  ├─ Saves to localStorage (meetingmind_history)
  └─ Auto-exports to Notion (if enabled)
```

---

## 11. External Integrations

### Notion Export
- **Module**: `backend/api/notion_export.py`
- Creates formatted Notion pages with headings, tables, bullet lists, callouts, and dividers
- Supports automatic export on analysis completion (toggle in Settings)
- Handles Notion's 100-block-per-request limit with batch appending
- **Endpoints**: `POST /api/export_notion`, `GET /api/notion_status`

### Google Translate
- Frontend embeds the Google Translate widget for full UI translation
- CSS overrides hide the default Google Translate banner to maintain UI aesthetics
- Separate from the backend language support (which generates AI content natively in the target language)

---

## 12. Configuration & Environment

### Environment Variables (`.env`)
| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `NOTION_API_KEY` | Notion integration token |
| `NOTION_DATABASE_ID` | Target Notion database ID |

### Central Config (`config.py`)
- Label set: `DECISION`, `ACTION_ITEM`, `DEADLINE`, `DISCUSSION`, `OTHER`
- TF-IDF max features: 5,000 (config default; vectorizer uses 15,000)
- Upload: Max 10 MB; PDF, DOC, DOCX, TXT supported
- spaCy model: `en_core_web_sm`
- Transformer settings: DistilBERT (reserved for future DL model)

---

## 13. How to Run

### Prerequisites
- Python 3.10+
- Trained ML model artifacts in `backend/ml_model/models/saved/`

### Installation
```bash
pip install -r requirements.txt
```

### Start the Backend (FastAPI)
```bash
python -m uvicorn backend.api.analysis_server:app --port 8502 --reload
```

### Start the Frontend (Streamlit)
```bash
streamlit run app.py
```

### Access
- **Dashboard**: http://localhost:8501
- **API docs**: http://localhost:8502/docs

---

*Last updated: May 2026*
