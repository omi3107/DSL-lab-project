import json
import csv
import glob
import os
import re

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

def contains_keywords(sentence, keywords):
    sentence_lower = sentence.lower()
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, sentence_lower):
            return True
    return False

def get_label(sentence):
    sentence_lower = sentence.lower()
    
    # 1. Edge case: "decided" -> always decision
    if re.search(r'\bdecided\b', sentence_lower):
        return "Decision"
    
    has_decision = contains_keywords(sentence, DECISION_KEYWORDS)
    has_task = contains_keywords(sentence, TASK_KEYWORDS)
    has_deadline = contains_keywords(sentence, DEADLINE_KEYWORDS)
    has_issue = contains_keywords(sentence, ISSUE_KEYWORDS)
    
    # "If sentence mentions problem -> issue"
    if has_issue:
        return "Issue"
        
    # Decision comes next
    if has_decision:
        return "Decision"
        
    # "If sentence has task + deadline -> label = task"
    if has_task and has_deadline:
        return "Task"
        
    if has_task:
        return "Task"
        
    if has_deadline:
        return "Deadline"
        
    return "General"

def parse_json_stream(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
            data.append(obj)
            idx = end_idx
        except Exception as e:
            print(f"Error parsing JSON from {filepath} at index {idx}: {e}")
            break
    return data

def main():
    root_dir = r"c:\Users\omini\Downloads\MeetingMind-AI-Redesigned\MeetingMind-AI-Redesigned"
    data_dir = os.path.join(root_dir, "github_raw_data", "GoogleData")
    output_csv = os.path.join(root_dir, "ml_backend", "dataset", "labelled_data.csv")

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print("Generating dataset...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'sentence', 'label', 'source'])
        
        for jf in json_files:
            source = os.path.basename(jf)
            print(f"Processing {source}...")
            data = parse_json_stream(jf)
            
            for meeting_obj in data:
                dialog_id = meeting_obj.get("dialogId", "unknown")
                meeting = meeting_obj.get("meeting", {})
                meeting_id = meeting.get("meetingId", "unknown")
                segments = meeting.get("transcriptSegments", [])
                
                for idx, segment in enumerate(segments):
                    text = segment.get("text", "").strip()
                    if not text:
                        continue
                    
                    label = get_label(text)
                    unique_id = f"{dialog_id}_{meeting_id}_{idx}"
                    
                    writer.writerow([unique_id, text, label, source])

    print(f"Dataset generated and saved to {output_csv}")

if __name__ == '__main__':
    main()
