"""
analysis_server.py — FastAPI backend for MeetingMind.

Pipeline:
    1. Receive raw transcript text from the frontend
    2. Run trained SVM classifier (insight_extractor) for sentence-level
       classification into: Decision · Task · Deadline · Issue · General
    3. Pass ML-classified results to Gemini/Groq AI layer for reframing:
       - Clean bullet points
       - Participant detection
       - Responsibility mapping
       - Intelligence score & tags
       - One-line summary
    4. Return structured JSON to the frontend
"""

import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

# ── Path setup ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "backend" / "ml_model") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend" / "ml_model"))

from src.insight_extractor import extract_insights
from src.gemini_layer import refine_insights
from backend.api.notion_export import export_to_notion, check_notion_connection

logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(title="MeetingMind Analysis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyseRequest(BaseModel):
    text: str


class AnalyseResponse(BaseModel):
    participants: list = []
    decisions: list = []
    tasks: list = []
    deadlines: list = []
    issues: list = []
    general: list = []
    score: int = 0
    confidence: int = 0
    tags: list = []
    title: str = ""
    responsibility_map: dict = {}
    ai_provider: str = ""


@app.post("/api/analyse", response_model=AnalyseResponse)
async def analyse_meeting(req: AnalyseRequest):
    """Run the full ML + AI pipeline on a transcript."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty transcript provided.")

    # ── Step 1: ML Classification ────────────────────────
    try:
        raw_insights = extract_insights(req.text)
    except Exception as e:
        logger.error("ML classification failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ML model error: {e}")

    if raw_insights.get("errors"):
        logger.warning("ML extraction warnings: %s", raw_insights["errors"])

    # ── Step 2: AI Reframing (Gemini/Groq) ───────────────
    try:
        refined = refine_insights(raw_insights)
    except Exception as e:
        logger.error("AI reframing failed: %s", e, exc_info=True)
        # Fall back to raw ML results if AI fails
        refined = _format_raw_as_response(raw_insights)

    # ── Step 3: Build response ───────────────────────────
    response = _build_response(raw_insights, refined)
    return response

@app.post("/api/extract_text")
async def extract_text_from_file(file: UploadFile = File(...)):
    """Extract text from uploaded PDF, DOCX, or TXT file."""
    try:
        content = await file.read()
        text = ""
        filename = file.filename.lower()
        
        if filename.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif filename.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF, DOCX, or TXT.")
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in the file.")
            
        return {"text": text.strip()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File extraction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {e}")


def _build_response(raw_insights: dict, refined: dict) -> dict:
    """Merge ML + AI results into the frontend-expected format."""

    # Tasks: AI layer returns [{task, assignee, deadline}]
    # Frontend expects [{who, task, by}]
    tasks_raw = refined.get("tasks", [])
    tasks = []
    for t in tasks_raw:
        if isinstance(t, dict):
            tasks.append({
                "who": t.get("assignee", t.get("who", "Unassigned")),
                "task": t.get("task", str(t)),
                "by": t.get("deadline", t.get("by", None)),
            })
        else:
            tasks.append({"who": "Unassigned", "task": str(t), "by": None})

    # Deadlines: AI returns strings or dicts
    deadlines_raw = refined.get("deadlines", [])
    deadlines = []
    for d in deadlines_raw:
        if isinstance(d, dict):
            deadlines.append(d.get("description", str(d)))
        else:
            deadlines.append(str(d))

    # Intelligence score
    intel = refined.get("intelligence_score", {})
    if isinstance(intel, dict):
        score = intel.get("score", 0)
    else:
        score = raw_insights.get("intelligence_score", {}).get("score", 0)

    # Confidence from ML model average
    confidence = _calc_avg_confidence(raw_insights)

    # Tags from AI assessment
    tags = _generate_tags(raw_insights, refined, score)

    # General discussion
    general = refined.get("general_discussion", refined.get("general", []))

    return {
        "participants": refined.get("participants", []),
        "decisions": refined.get("decisions", []),
        "tasks": tasks,
        "deadlines": deadlines,
        "issues": refined.get("issues", []),
        "general": general,
        "score": score,
        "confidence": confidence,
        "tags": tags,
        "title": refined.get("meeting_title", "Meeting Analysis"),
        "responsibility_map": refined.get("responsibility_map", {}),
        "ai_provider": refined.get("_ai_provider", ""),
    }


def _format_raw_as_response(raw: dict) -> dict:
    """Format raw ML output when AI reframing fails."""
    return {
        "meeting_title": "Meeting Analysis",
        "participants": [],
        "decisions": raw.get("decisions", []),
        "tasks": raw.get("tasks", []),
        "deadlines": [
            d.get("description", str(d)) if isinstance(d, dict) else str(d)
            for d in raw.get("deadlines", [])
        ],
        "issues": raw.get("issues", []),
        "general_discussion": raw.get("general", []),
        "responsibility_map": {},
        "intelligence_score": raw.get("intelligence_score", {"score": 0}),
        "_ai_provider": "None (ML only)",
    }


def _calc_avg_confidence(raw: dict) -> int:
    """Calculate average ML model confidence across all predictions."""
    # The raw_insights doesn't store per-sentence confidence directly,
    # so we estimate from the intelligence score
    intel = raw.get("intelligence_score", {})
    score = intel.get("score", 0) if isinstance(intel, dict) else 0
    # Map score to a confidence range (60-95)
    return min(95, max(60, int(score * 0.8 + 20)))


def _generate_tags(raw: dict, refined: dict, score: int) -> list:
    """Generate quality tags based on ML results."""
    tags = []

    n_dec = len(refined.get("decisions", []))
    n_task = len(refined.get("tasks", []))
    n_iss = len(refined.get("issues", []))
    n_dl = len(refined.get("deadlines", []))

    # Structure tag
    if n_dec > 0 and n_task > 0:
        tags.append("Structured")
    elif n_dec == 0 and n_task == 0:
        tags.append("Unstructured")

    # Action density
    if n_task >= 4:
        tags.append("Action-Heavy")
    elif n_task >= 2:
        tags.append(f"{n_task} Actions")
    elif n_task == 0:
        tags.append("No Actions")

    # Risk assessment
    if n_iss >= 3:
        tags.append("High Risk")
    elif n_iss > 0:
        tags.append("Has Issues")

    # Deadline awareness
    if n_dl >= 2:
        tags.append("Deadline-Driven")

    # Productivity
    if score >= 70:
        tags.append("High Stakes")
    elif score < 30:
        tags.append("Low Impact")

    return tags[:4]  # Cap at 4 tags


# ── Notion endpoints ─────────────────────────────────────
class NotionExportRequest(BaseModel):
    meeting_data: dict


@app.post("/api/export_notion")
async def export_notion_endpoint(req: NotionExportRequest):
    """Export meeting insights to Notion as a formatted page."""
    try:
        result = await export_to_notion(req.meeting_data)
        return {"status": "ok", "url": result["url"], "page_id": result["page_id"]}
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Notion export failed: %s", e, exc_info=True)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Notion export failed: {e}")


@app.get("/api/notion_status")
async def notion_status():
    """Check if Notion credentials are configured and valid."""
    result = await check_notion_connection()
    return result


# ── Health check ─────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline": "ML (SVM) + AI (Gemini/Groq)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
