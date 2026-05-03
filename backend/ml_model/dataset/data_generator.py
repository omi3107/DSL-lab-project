import csv
import random
import re
import os
import time
import requests
from pathlib import Path
from collections import Counter

import pandas as pd
from dotenv import load_dotenv, find_dotenv

# ============================================================
# Config
# ============================================================
# Load environment variables (robustly find .env in root)
load_dotenv(find_dotenv())

CSV_PATH = Path(__file__).resolve().parent / "labelled_data.csv"
TARGET_PER_CLASS = 25_000
SOURCE_TAG = "synthetic_llm_aug_v1"
RANDOM_SEED = 42
LLM_MODEL_NAME = "llama-3.3-70b-versatile" # Groq model
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_RATIO = 0.20                         # Reduced to be more rate-limit friendly
MAX_LLM_TRIES = 2

random.seed(RANDOM_SEED)

# Global rate limit state
GLOBAL_COOLDOWN_UNTIL = 0  # Timestamp until which LLM calls are skipped
COOLDOWN_DURATION = 60    # Seconds to wait after a 429
SUCCESS_DELAY = 0.5       # Seconds to wait after a successful API call to avoid burst limits

# ============================================================
# Label keywords (same spirit as your guideline)
# ============================================================
DECISION_KEYWORDS = [
    "decided", "agree", "agreed", "agreement", "finalized", "finalised",
    "confirmed", "conclude", "concluded", "conclusion", "approved",
    "approve", "approval", "settled", "locked", "locked in", "chosen",
    "selected", "selection", "determined", "resolved", "resolution",
    "committed", "commitment", "go with", "go ahead with", "proceed with",
    "sign off", "signed off", "greenlight", "greenlit", "endorsed",
    "accepted", "accept", "ratified"
]

TASK_KEYWORDS = [
    "will", "should", "shall", "need to", "needs to", "must", "have to",
    "has to", "assign", "assigned", "assigning", "take up", "handle",
    "work on", "prepare", "create", "build", "develop", "design", "fix",
    "update", "review", "check", "complete", "finish", "deliver", "send",
    "share", "coordinate", "follow up", "follow-up", "arrange", "schedule",
    "plan", "draft", "implement", "test", "deploy", "investigate",
    "analyze", "document"
]

DEADLINE_KEYWORDS = [
    "by", "before", "due", "deadline", "end of day", "eod", "end of week",
    "eow", "end of month", "eom", "tomorrow", "today", "tonight",
    "next week", "next month", "this week", "this month", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "date", "time", "at", "on", "within", "no later than", "latest by",
    "target date", "target time", "scheduled for", "schedule by",
    "complete by", "submit by", "finish by"
]

ISSUE_KEYWORDS = [
    "pending", "unclear", "not finalized", "not finalised", "not decided",
    "undecided", "issue", "problem", "concern", "risk", "blocker",
    "blocked", "blocking", "dependency", "challenge", "difficulty",
    "confusion", "ambiguity", "uncertain", "uncertainty", "incomplete",
    "missing", "gap", "error", "bug", "failure", "failing", "stuck",
    "delay", "delayed", "unresolved", "open issue", "open question",
    "needs clarification", "requires clarification", "needs discussion", 
    "requires discussion"
]

# ============================================================
# Helpers
# ============================================================
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def contains_keywords(sentence: str, keywords: list[str]) -> bool:
    s = sentence.lower()
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, s):
            return True
    return False

def infer_label(sentence: str) -> str:
    s = sentence.lower()

    # guideline priority
    if re.search(r"\bdecided\b", s):
        return "Decision"

    has_issue = contains_keywords(sentence, ISSUE_KEYWORDS)
    has_decision = contains_keywords(sentence, DECISION_KEYWORDS)
    has_task = contains_keywords(sentence, TASK_KEYWORDS)
    has_deadline = contains_keywords(sentence, DEADLINE_KEYWORDS)

    if has_issue:
        return "Issue"
    if has_decision:
        return "Decision"
    if has_task and has_deadline:
        return "Task"
    if has_task:
        return "Task"
    if has_deadline:
        return "Deadline"
    return "General"

def first_sentence(text: str) -> str:
    text = str(text).strip()
    # Remove markers like 1., * or -
    text = re.sub(r"^\s*[\-\*\d\.\)\:]+\s*", "", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    text = parts[0].strip() if parts and parts[0].strip() else text
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text

def is_good_candidate(sentence: str, label: str, seen: set[str]) -> bool:
    if not sentence or len(sentence.split()) < 4:
        return False
    if norm(sentence) in seen:
        return False
    return infer_label(sentence) == label

# ============================================================
# Rule-based seed generation
# ============================================================
def make_seed(label: str) -> str:
    if label == "Decision":
        actors = ["we", "the team", "the committee", "the client", "the leads", "the group", "management", "the board", "stakeholders", "the engineering team", "the product team", "the marketing team", "the executive group", "the QA team"]
        verbs = ["decided", "agreed", "finalized", "confirmed", "approved", "locked in", "selected", "resolved", "agreed upon", "greenlit", "signed off on", "settled on", "committed to", "chose"]
        objects = [
            "the release plan", "the API approach", "the final design", "the demo date",
            "the feature scope", "the testing flow", "the deployment strategy",
            "the presentation format", "the project direction", "the meeting schedule",
            "the budget allocation", "the hiring plan", "the technical stack",
            "the new architecture", "the UI mockups", "the launch strategy",
            "the go-to-market plan", "the pricing tier", "the vendor selection",
            "the security protocol", "the database schema", "the MVP requirements"
        ]
        extras = [
            "for this sprint", "for the current release", "after the latest feedback",
            "during the review", "for the next phase", "for the demo", "effective immediately",
            "moving forward", "for Q3", "for Q4", "until further notice", "to meet the deadline",
            "as per the client request", "based on the pilot feedback"
        ]
        templates = [
            "{actor} {verb} {object} {extra}.",
            "After discussion, {actor} {verb} {object}.",
            "The {object} was {verb} {extra}.",
            "{actor_cap} {verb} {object} {extra}.",
            "We have {verb} {object} {extra}."
        ]
        actor = random.choice(actors)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        extra = random.choice(extras)
        template = random.choice(templates)
        return template.format(actor=actor, actor_cap=actor.capitalize(), verb=verb, object=obj, extra=extra)

    if label == "Task":
        people = ["Rahul", "Aisha", "Neha", "Aman", "the developer", "the designer", "the analyst", "the team", "the manager", "Priya", "Vikram", "Sarah", "John", "David", "Emma", "the external vendor", "the QA engineer", "the product owner", "the scrum master"]
        actions = ["prepare", "review", "update", "fix", "document", "send", "share", "build", "test", "draft", "schedule", "coordinate", "follow up on", "investigate", "analyze", "implement", "deploy", "audit", "refactor", "monitor"]
        objects = [
            "the slides", "the report", "the bug fix", "the checklist", "the notes",
            "the prototype", "the summary", "the workflow", "the test cases", "the deployment steps",
            "the API documentation", "the user manual", "the analytics dashboard", "the backend scripts",
            "the frontend components", "the design assets", "the client emails", "the contract",
            "the onboarding guide", "the release notes"
        ]
        times = ["today", "by tomorrow", "by Friday", "this week", "after lunch", "before the review", "soon", "by End of Day", "by next Monday", "within 24 hours", "asap", "before the end of the sprint", "by the end of the month"]
        extras = [
            "for the meeting", "for the client", "for the next sprint", "for the demo",
            "before the deadline", "for the final review", "to unblock the team", "for the presentation",
            "to ensure compliance", "based on the new requirements", "as a top priority"
        ]
        templates = [
            "{person} will {action} {object} {time} {extra}.",
            "{person} should {action} {object} {extra}.",
            "Please {action} {object} {time}.",
            "{person} needs to {action} {object} {time}.",
            "The team must {action} {object} {extra}.",
            "{person} is going to {action} {object} {time}."
        ]
        return random.choice(templates).format(
            person=random.choice(people),
            action=random.choice(actions),
            object=random.choice(objects),
            time=random.choice(times),
            extra=random.choice(extras)
        )

    if label == "Deadline":
        departments = ["marketing", "sales", "engineering", "HR", "design", "QA", "backend", "frontend", "infrastructure", "legal", "finance", "operations"]
        items = ["submission", "review", "demo", "feedback", "deployment", "presentation", "prototype", "testing", "report", "audit", "documentation", "analysis", "sprint plan", "user study", "budget approval", "security patch", "mockup", "performance test"]
        verbs = ["is due", "must be completed", "should be finished", "needs to be submitted", "has to be ready", "will be required", "must be concluded", "needs finalization"]
        times = ["by Monday", "by Tuesday", "by Wednesday", "by Thursday", "by Friday", "tomorrow", "today", "by end of day", "by next week", "before 5 PM", "by the deadline", "by EOD", "before the weekend", "by the end of the month"]
        extras = [
            "for the project", "for the client", "for the final round", "for the release",
            "for the current sprint", "for the meeting", "for the MVP", "for the v2.0 update",
            "for compliance", "for management", "as per the new schedule"
        ]
        templates = [
            "The {item} {verb} {time} {extra}.",
            "{item_cap} {verb} {time}.",
            "Please complete the {item} {time} for {dept}.",
            "The final {item} is scheduled {time} {extra}.",
            "{item_cap} {verb} {time} {extra}."
        ]
        item = random.choice(items)
        return random.choice(templates).format(
            item=item,
            item_cap=item.capitalize(),
            dept=random.choice(departments),
            verb=random.choice(verbs),
            time=random.choice(times),
            extra=random.choice(extras)
        )

    if label == "Issue":
        subjects = ["pricing", "timeline", "feature scope", "deployment", "login flow", "data export", "testing", "requirement", "authentication", "UI design", "database migration", "API integration", "payment gateway", "onboarding process", "accessibility compliance", "security audit", "load balancing", "server configuration", "third-party integration", "billing system"]
        states = ["is unclear", "is blocked", "is delayed", "needs clarification", "is not finalized", "is causing issues", "is still pending", "has a blocker", "presents a challenge", "is throwing errors", "remains ambiguous", "needs a rethink", "is entirely blocked", "is proving problematic"]
        causes = [
            "right now", "at the moment", "for this release", "because of testing",
            "due to a dependency", "from the backend team", "in the current version",
            "due to resource constraints", "because of lack of data", "since yesterday",
            "due to API changes", "from the client side", "in the latest build",
            "because of conflicting requirements", "due to an upstream bug"
        ]
        prefixes = ["", "Unfortunately, ", "As you know, ", "Just flagging that ", "Note that ", "One concern: ", "We identified that ", "It appears ", "Currently, ", "We noticed that ", "Quick update: "]
        templates = [
            "{prefix}the {subject} {state} {cause}.",
            "{prefix}there is an issue with {subject} {cause}.",
            "{subject_cap} {state}.",
            "We are still stuck on {subject} {cause}.",
            "The {subject} remains unresolved {cause}."
        ]
        subject = random.choice(subjects)
        prefix = random.choice(prefixes)
        
        # fix capitalization for prefixes if needed
        template = random.choice(templates)
        if template.startswith("{prefix}") and prefix:
            prefix = prefix.capitalize() if prefix[0].islower() else prefix
            
        res = template.format(
            prefix=prefix,
            subject=subject,
            subject_cap=subject.capitalize() if not prefix else subject,
            state=random.choice(states),
            cause=random.choice(causes)
        )
        return res

    return ""

# ============================================================
# Gemini API based rewrite (using new google.genai SDK)
# ============================================================
def build_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[WARN] GROQ_API_KEY not found in environment.")
        return None
    
    # We return the API key to be used by requests
    return api_key

def llm_rewrite(label: str, seed: str, api_key: str) -> list[str]:
    global GLOBAL_COOLDOWN_UNTIL
    
    if not api_key:
        return []
        
    current_time = time.time()
    if current_time < GLOBAL_COOLDOWN_UNTIL:
        # Silently skip during cooldown to avoid hitting the API
        return []

    label_rules = {
        "Decision": "finalized or agreed meeting decision.",
        "Task": "assigned action item for a person or team.",
        "Deadline": "time constraint, due date, or deadline.",
        "Issue": "unresolved problem, blocker, concern, or clarification need."
    }

    prompt = f"""
I need 10 natural, diverse meeting transcript sentences that represent a {label}.
Type: {label_rules[label]}
Rule: Sound like real people talking in a business meeting. No labels. One sentence per line.
Variations should be based on this core idea: {seed}
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a professional meeting transcript generator. Output ONLY 10 unique sentences, one per line. No numbers, no labels."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 500
    }

    outputs = []
    max_retries = 2 # Reduced retries because we have global cooldown now

    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                res_json = response.json()
                content = res_json['choices'][0]['message']['content']
                if content:
                    lines = content.strip().split("\n")
                    for line in lines:
                        cleaned = first_sentence(line)
                        if cleaned and len(cleaned.split()) >= 4:
                            outputs.append(cleaned)
                
                # Small delay to respect burst limits
                time.sleep(SUCCESS_DELAY)
                break
            elif response.status_code == 429:
                print(f"[INFO] Groq Rate limit (429) hit. Entering global cooldown for {COOLDOWN_DURATION}s...")
                GLOBAL_COOLDOWN_UNTIL = time.time() + COOLDOWN_DURATION
                break
            else:
                print(f"[WARN] Groq API returned status {response.status_code}: {response.text[:200]}")
                break
        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}")
            break

    return outputs

# ============================================================
# Main augmentation
# ============================================================
def augment_label(label: str, target_count: int, existing_df: pd.DataFrame, llm_key: str) -> list[dict]:
    current = int((existing_df["label"] == label).sum())
    needed = max(0, target_count - current)
    if needed == 0:
        print(f"{label}: already at target or above.")
        return []

    print(f"{label}: need {needed} more rows")

    existing_label_sentences = existing_df.loc[
        existing_df["label"] == label, "sentence"
    ].astype(str).tolist()

    seen = {norm(s) for s in existing_label_sentences}
    new_rows = []
    next_num = len(existing_df) + 1

    attempts = 0
    max_attempts = needed * 30 # Increased attempts for variety

    while len(new_rows) < needed and attempts < max_attempts:
        attempts += 1

        seed = make_seed(label)
        if not seed:
            continue

        candidates = []
        # Try LLM rewrite based on ratio
        if llm_key is not None and random.random() < LLM_RATIO:
            candidates.extend(llm_rewrite(label, seed, llm_key))

        # Always include seed as a fallback or if LLM skipped/exhausted
        candidates.append(seed)

        added_in_this_step = 0
        for cand in candidates:
            if len(new_rows) >= needed:
                break
            cand = first_sentence(cand)
            if is_good_candidate(cand, label, seen):
                row = {
                    "id": f"aug_{next_num}",
                    "sentence": cand,
                    "label": label,
                    "source": SOURCE_TAG
                }
                new_rows.append(row)
                seen.add(norm(cand))
                next_num += 1
                added_in_this_step += 1

        if added_in_this_step > 0 and (len(new_rows) % 100 < added_in_this_step or len(new_rows) % 100 == 0):
            print(f"  {label}: generated {len(new_rows)}/{needed}")

    print(f"{label}: added {len(new_rows)}/{needed}")
    return new_rows

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, quotechar='"', encoding='utf-8')
    required_cols = {"id", "sentence", "label", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required_cols)}")

    print("Current counts:")
    print(df["label"].value_counts().to_string())
    print()

    llm_key = build_llm()
    if llm_key is None:
        print("[INFO] Running in rule-based mode only (No Groq API).")
    else:
        print(f"[INFO] Groq API Session loaded: {LLM_MODEL_NAME} (REST API Mode)")

    all_new_rows = []
    # Process minority classes
    for label in ["Decision", "Task", "Deadline", "Issue"]:
        rows = augment_label(label, TARGET_PER_CLASS, df, llm_key)
        all_new_rows.extend(rows)

    if not all_new_rows:
        print("Nothing to append.")
        return

    # Append to CSV
    with CSV_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sentence", "label", "source"])
        for row in all_new_rows:
            writer.writerow(row)

    # Final report
    updated = pd.read_csv(CSV_PATH)
    print("\nUpdated counts:")
    print(updated["label"].value_counts().to_string())
    print(f"\nAppended {len(all_new_rows):,} rows to {CSV_PATH}")

if __name__ == "__main__":
    main()