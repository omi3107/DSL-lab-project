from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

router = APIRouter()

class EntitiesRequest(BaseModel):
    text: str

# Initialize NER pipeline
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ner_pipe = pipeline(
        "ner", 
        model="dbmdz/bert-large-cased-finetuned-conll03-english", 
        aggregation_strategy="simple",
        device=device
    )
except Exception as e:
    print(f"Warning: Could not load NER pipeline: {e}")
    ner_pipe = None

@router.post("/extract_entities")
async def extract_entities(req: EntitiesRequest):
    """Extract named entities from text using BERT (CO-3)."""
    if ner_pipe is None:
        raise HTTPException(status_code=503, detail="NER model not initialized")

    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        entities = ner_pipe(req.text)
        
        formatted_entities = [
            {
                "entity_group": ent["entity_group"],
                "score": float(ent["score"]),
                "word": ent["word"],
                "start": ent["start"],
                "end": ent["end"]
            }
            for ent in entities
        ]

        return {"entities": formatted_entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")
