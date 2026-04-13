from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

router = APIRouter()

class SummarizeRequest(BaseModel):
    text: str

# Initialize summarization pipeline
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    summarizer_pipe = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn", 
        device=device
    )
except Exception as e:
    print(f"Warning: Could not load BART summarizer pipeline: {e}")
    summarizer_pipe = None

@router.post("/summarize")
async def summarize_text(req: SummarizeRequest):
    """Summarize a meeting transcript using BART (CO-3)."""
    if summarizer_pipe is None:
        raise HTTPException(status_code=503, detail="Summarizer model not initialized")

    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        summary = summarizer_pipe(req.text, max_length=150, min_length=40, do_sample=False)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")
