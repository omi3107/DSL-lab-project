import sys
import os
from pathlib import Path

# Add the ml_backend directory to the sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import transcribe, summarize, entities
from preprocessing.text_cleaner import TextCleaner
from pydantic import BaseModel

app = FastAPI(
    title="MeetingMind ML Backend",
    description="Backend API fulfilling CO-3 and CO-4 constraints using ML/DL and NLP models.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Route
@app.get("/")
def read_root():
    return {"status": "ok", "message": "MeetingMind ML Backend is running"}

# Include ML Routes (CO-3)
app.include_router(transcribe.router, prefix="/api/v1", tags=["Speech Recognition"])
app.include_router(summarize.router, prefix="/api/v1", tags=["Summarization"])
app.include_router(entities.router, prefix="/api/v1", tags=["Named Entity Recognition"])

# Basic NLP Route (CO-4)
class CleanTextRequest(BaseModel):
    text: str

cleaner = TextCleaner()

@app.post("/api/v1/clean_text", tags=["NLP Preprocessing"])
async def clean_text(req: CleanTextRequest):
    """Cleans raw text using NLP preprocessing (CO-4)."""
    if not req.text or len(req.text.strip()) == 0:
        return {"cleaned_text": "", "keywords": []}
        
    cleaned = cleaner.clean_text(req.text)
    keywords = cleaner.extract_keywords(req.text)
    
    return {
        "original_text_length": len(req.text),
        "cleaned_text": cleaned,
        "cleaned_text_length": len(cleaned),
        "top_keywords": keywords
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
