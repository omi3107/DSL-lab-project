from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import tempfile
import os
import torch
from transformers import pipeline

router = APIRouter()

# Initialize ASR pipeline
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    whisper_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=device
    )
except Exception as e:
    print(f"Warning: Could not load Whisper pipeline: {e}")
    whisper_pipe = None


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file using Whisper (CO-3)."""
    if whisper_pipe is None:
        raise HTTPException(status_code=503, detail="Whisper model not initialized")

    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        result = whisper_pipe(tmp_path)
        os.unlink(tmp_path)

        return {
            "filename": file.filename,
            "text": result["text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
