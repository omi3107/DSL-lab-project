"""
gemini_layer.py — AI Framing Layer for MeetingMind.

Uses Gemini (primary) or Groq (fallback) to:
1. Refine ML-classified sentences into clean, concise bullet points
2. Extract responsibility mapping (participant → assigned tasks)
3. Generate meeting title and participant list
"""

import json
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Gemini setup ─────────────────────────────────────────
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error("Failed to initialize Gemini client: %s", e)
        GEMINI_AVAILABLE = False

# ── Groq setup ──────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ─────────────────────────────────────────────────────────
#  Prompt Builder
# ─────────────────────────────────────────────────────────

def _build_prompt(raw_insights: dict) -> str:
    """Build the AI framing prompt from ML-classified data."""
    prompt_data = {
        "decisions": raw_insights.get("decisions", []),
        "tasks": raw_insights.get("tasks", []),
        "deadlines": raw_insights.get("deadlines", []),
        "issues": raw_insights.get("issues", []),
        "general": raw_insights.get("general", []),
    }

    transcript = raw_insights.get("raw_transcript", "")
    snippet = transcript[:4000] if len(transcript) > 4000 else transcript

    return f"""You are an expert meeting analyst. You are given ML-classified meeting transcript data where each sentence has been categorised into: Decision, Task, Deadline, Issue, or General.

Your job is to refine this data into clean, professional, display-ready content.

**IMPORTANT: Return ONLY valid JSON. No markdown fences, no extra text.**

Return a JSON object with this EXACT structure:
{{
    "meeting_title": "A concise, descriptive title for this meeting",
    "participants": ["list", "of", "participant", "names"],
    "decisions": ["Refined bullet point 1", "Refined bullet point 2"],
    "tasks": [
        {{"task": "Clean description of the task", "assignee": "Person name or Unassigned", "deadline": "Deadline if mentioned, else null"}}
    ],
    "deadlines": ["Clean deadline description 1"],
    "issues": ["Clean issue description 1"],
    "general_discussion": ["Key discussion point 1"],
    "responsibility_map": {{
        "Person Name": ["task 1 they own", "task 2 they own"]
    }},
    "intelligence_score": {{
        "score": 75,
        "assessment": "Brief one-line assessment of meeting productivity"
    }}
}}

Rules:
1. Each bullet point must be ONE concise, clear sentence.
2. Extract real participant names from the transcript. If "Speaker N" is used, keep that format.
3. For responsibility_map, map each participant to their assigned tasks. Use "Unassigned" for tasks without a clear owner.
4. Remove filler words, clean up grammar, make each point actionable and clear.
5. If a section has no items, return an empty list [].
6. intelligence_score.score: 0 = unproductive, 100 = highly productive.
7. Do NOT invent information not present in the data.

ML-Classified Data:
```json
{json.dumps(prompt_data, indent=2)}
```

Original Transcript (for context and participant names):
\"\"\"
{snippet}
\"\"\"

Return ONLY the JSON object:"""


# ─────────────────────────────────────────────────────────
#  API Callers
# ─────────────────────────────────────────────────────────

def _clean_json_response(text: str) -> str:
    """Strip markdown code fences from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.lstrip().startswith("json"):
        text = text.lstrip()[4:]
    return text.strip()


def _call_gemini(prompt: str) -> dict | None:
    """Try Gemini API."""
    if not gemini_client:
        logger.info("Gemini client not initialized")
        return None
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        cleaned = _clean_json_response(response.text)
        return json.loads(cleaned)
    except Exception as e:
        logger.warning("Gemini API failed: %s", e)
        return None


def _call_groq(prompt: str) -> dict | None:
    """Try Groq API as fallback."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        logger.info("Groq not available (missing library or API key)")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a meeting analysis AI. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        cleaned = _clean_json_response(response.choices[0].message.content)
        return json.loads(cleaned)
    except Exception as e:
        logger.warning("Groq API failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────
#  Fallback Formatter (no AI)
# ─────────────────────────────────────────────────────────

def _fallback_format(raw_insights: dict) -> dict:
    """Format insights without AI when both APIs fail."""
    return {
        "meeting_title": "Meeting Analysis",
        "participants": [],
        "decisions": raw_insights.get("decisions", []),
        "tasks": raw_insights.get("tasks", []),
        "deadlines": [
            d.get("description", str(d))
            for d in raw_insights.get("deadlines", [])
        ],
        "issues": raw_insights.get("issues", []),
        "general_discussion": raw_insights.get("general", []),
        "responsibility_map": {},
        "intelligence_score": raw_insights.get(
            "intelligence_score",
            {"score": 0, "assessment": "AI framing unavailable"},
        ),
    }


# ─────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────

def refine_insights(raw_insights: dict) -> dict:
    """Refine raw ML-classified insights using AI.

    Tries Groq first, then Gemini, then falls back to basic formatting.

    Returns:
        Structured dict ready for UI display.
    """
    prompt = _build_prompt(raw_insights)

    # 1. Try Groq (primary)
    result = _call_groq(prompt)
    if result:
        logger.info("AI framing completed via Groq")
        result["_ai_provider"] = "Groq"
        return result

    # 2. Fallback to Gemini
    result = _call_gemini(prompt)
    if result:
        logger.info("AI framing completed via Gemini")
        result["_ai_provider"] = "Gemini"
        return result

    # 3. Both failed — basic formatting
    logger.warning("Both Gemini and Groq failed. Using basic formatting.")
    fallback = _fallback_format(raw_insights)
    fallback["_ai_provider"] = "None (fallback)"
    return fallback


# ─────────────────────────────────────────────────────────
#  AI Summary Generation
# ─────────────────────────────────────────────────────────

def _build_summary_prompt(refined: dict) -> str:
    """Build a prompt for generating a structured meeting summary."""
    sections = []
    if refined.get("meeting_title"):
        sections.append(f"Meeting Title: {refined['meeting_title']}")
    if refined.get("participants"):
        sections.append(f"Participants: {', '.join(refined['participants'])}")
    if refined.get("decisions"):
        sections.append(f"Decisions: {json.dumps(refined['decisions'])}")
    if refined.get("tasks"):
        sections.append(f"Tasks: {json.dumps(refined['tasks'])}")
    if refined.get("deadlines"):
        sections.append(f"Deadlines: {json.dumps(refined['deadlines'])}")
    if refined.get("issues"):
        sections.append(f"Issues: {json.dumps(refined['issues'])}")
    if refined.get("general_discussion"):
        sections.append(f"General Discussion: {json.dumps(refined['general_discussion'])}")
    intel = refined.get("intelligence_score", {})
    if isinstance(intel, dict) and intel.get("score"):
        sections.append(f"Intelligence Score: {intel['score']}/100 — {intel.get('assessment', '')}")

    data_block = "\n".join(sections)

    return f"""You are an expert meeting analyst. Generate a clear, structured executive summary from these meeting insights.

**RULES:**
1. Write in professional, concise language.
2. Only include sections that have data — skip empty sections entirely.
3. Do NOT invent information not present in the data.
4. Use markdown formatting with headers (##), bullet points, and bold text.
5. Keep it under 400 words.

**SECTIONS TO INCLUDE (only if data exists):**
- ## Meeting Overview — Title, participants, brief context
- ## Key Decisions — Numbered list
- ## Action Items — Tasks with assignees and deadlines (table format if possible)
- ## Risk Assessment — Issues and potential impact
- ## Timeline — Upcoming deadlines
- ## Productivity Assessment — Score and brief assessment

Meeting Data:
{data_block}

Generate the summary now:"""


def _call_gemini_text(prompt: str) -> str | None:
    """Call Gemini API and return raw text."""
    if not gemini_client:
        return None
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.warning("Gemini text call failed: %s", e)
        return None


def _call_groq_text(prompt: str) -> str | None:
    """Call Groq API and return raw text."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional meeting analyst. Write clear, structured summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Groq text call failed: %s", e)
        return None


def _fallback_summary(refined: dict) -> str:
    """Generate a basic summary without AI when both APIs fail."""
    parts = []
    title = refined.get("meeting_title", "Meeting Analysis")
    parts.append(f"## Meeting Overview\n**{title}**")

    participants = refined.get("participants", [])
    if participants:
        parts.append(f"**Participants:** {', '.join(participants)}")

    decisions = refined.get("decisions", [])
    if decisions:
        items = "\n".join(f"{i+1}. {d}" for i, d in enumerate(decisions))
        parts.append(f"\n## Key Decisions\n{items}")

    tasks = refined.get("tasks", [])
    if tasks:
        items = []
        for t in tasks:
            if isinstance(t, dict):
                assignee = t.get("assignee", "Unassigned")
                task = t.get("task", str(t))
                dl = f" (by {t['deadline']})" if t.get("deadline") else ""
                items.append(f"- **{assignee}** → {task}{dl}")
            else:
                items.append(f"- {t}")
        parts.append(f"\n## Action Items\n" + "\n".join(items))

    issues = refined.get("issues", [])
    if issues:
        items = "\n".join(f"- ⚠️ {i}" for i in issues)
        parts.append(f"\n## Risk Assessment\n{items}")

    deadlines = refined.get("deadlines", [])
    if deadlines:
        items = []
        for dl in deadlines:
            items.append(f"- {dl}" if isinstance(dl, str) else f"- {dl.get('description', str(dl))}")
        parts.append(f"\n## Timeline\n" + "\n".join(items))

    intel = refined.get("intelligence_score", {})
    if isinstance(intel, dict) and intel.get("score"):
        assessment = intel.get("assessment", "N/A")
        parts.append(f"\n## Productivity Assessment\n**Score:** {intel['score']}/100 — {assessment}")

    return "\n".join(parts)


def generate_ai_summary(refined: dict) -> str:
    """Generate a structured AI summary from refined meeting insights.

    Tries Gemini first, then Groq, then falls back to basic formatting.

    Returns:
        Markdown-formatted summary string.
    """
    prompt = _build_summary_prompt(refined)

    # 1. Try Gemini (primary)
    result = _call_gemini_text(prompt)
    if result:
        logger.info("AI summary generated via Gemini")
        return result

    # 2. Fallback to Groq
    result = _call_groq_text(prompt)
    if result:
        logger.info("AI summary generated via Groq")
        return result

    # 3. Both failed — basic formatting
    logger.warning("Both APIs failed for summary. Using fallback.")
    return _fallback_summary(refined)


if __name__ == "__main__":
    sample = {
        "decisions": ["We decided to go with React for the frontend."],
        "tasks": [{"assignee": "Unassigned", "task": "Rahul will prepare slides by Friday.", "deadline": None}],
        "deadlines": [{"description": "Submit report by end of week."}],
        "issues": ["API integration is still pending."],
        "general": ["Let's move on to the next topic."],
        "raw_transcript": "Rahul: We decided to go with React. Sarah: Rahul will prepare slides by Friday.",
    }
    result = refine_insights(sample)
    print(json.dumps(result, indent=2))
