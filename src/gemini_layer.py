"""
gemini_layer.py — Gemini AI Reframing Layer for MeetingMind.
Takes raw meeting insights and produces short, precise bullet points.
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def refine_insights(raw_insights: dict) -> str:
    """
    Takes the raw insights dictionary from insight_extractor and sends it
    to Gemini for reframing into short, precise bullet points.
    Returns a formatted markdown string.
    """
    if not GEMINI_AVAILABLE:
        return "⚠️ `google-generativeai` is not installed. Run: `pip install google-generativeai`"

    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured. Add `GEMINI_API_KEY` to your `.env` file."

    prompt = f"""You are an expert meeting analyst. Given the following raw meeting analysis JSON, 
produce a clean, professional summary formatted as short and precise bullet points.

Group the output into these sections:
1. **Meeting Overview** — Title, participants, estimated duration
2. **Key Decisions** — What was decided
3. **Action Items** — Who needs to do what, by when
4. **Deadlines** — Important dates
5. **Open Issues** — Unresolved items
6. **Intelligence Score** — Overall meeting effectiveness

Rules:
- Each bullet point must be ONE concise sentence.
- Use markdown formatting (bold for names/dates, bullet lists).
- If a section has no data, write "None identified."
- Do NOT add information that is not in the data.

Raw Analysis JSON:
```json
{json.dumps(raw_insights, indent=2)}
```

Provide the refined summary below:"""

    try:
        # Using gemini-1.5-flash for better free-tier stability
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Gemini API Quota Exceeded (429). Please check your Google AI Studio billing/plan."
        return f"⚠️ Gemini API error: {str(e)}"


if __name__ == "__main__":
    # Quick test
    sample = {
        "meeting_summary": {"title": "Sprint Planning", "participants": ["Alice", "Bob"]},
        "decisions": ["Move launch to Nov 15"],
        "action_items": [{"assignee": "Alice", "task": "Update roadmap", "deadline": "Nov 10"}],
        "deadlines": [{"description": "Launch by Nov 15"}],
        "open_issues": ["Budget approval pending"],
        "intelligence_score": {"score": 72, "flags": []},
    }
    print(refine_insights(sample))
