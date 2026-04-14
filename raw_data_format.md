# 🧠 1. github_raw_data/AMICorpusXML/data/ami-transcripts (TXT Files)

📁 Example: `EN2001a.transcript.txt`

## 🔹 Format: Plain Text (Unstructured)

```text
Okay. Does anyone want to see Steve's feedback...
Right. Not really...
We should probably prioritize our packages...
```

👉 Key points:

* ❌ No JSON structure
* ❌ No speaker separation (in your version)
* ❌ No metadata
* ✅ Just continuous conversation text
* ⚠️ Contains fillers, noise, broken sentences 

---

# 🧠 2. github_raw_data/GoogleData/

📁 Example: `train.json`, `test.json`, `validation.json`

## 🔹 Format: Nested JSON

```json
{
  "dialogId": "004ac02783ba442e8eeb307ea45ee97c",
  "meeting": {
    "meetingId": "Bmr019",
    "transcriptSegments": [
      {
        "text": "OK, we're on.",
        "speakerName": "Grad E"
      },
      {
        "text": "OK.",
        "speakerName": "Professor B"
      }
    ]
  }
}
```

👉 Key points:

* ✅ Structured conversation
* ✅ Sentence-level already (huge advantage)
* ✅ Speaker info available
* ⚠️ Each entry = **one sentence (mostly)** 

---

# 🧠 3. github_raw_data/GoogleData/ (Another Example Structure)

Same format, different meeting:

```json
{
  "dialogId": "0300047c747a4c6bb8902346d84d422a",
  "meeting": {
    "meetingId": "ES2004a",
    "transcriptSegments": [
      {
        "text": "Hmm hmm hmm.",
        "speakerName": "User Interface"
      },
      {
        "text": "Are we allowed to dim the lights?",
        "speakerName": "Project Manager"
      }
    ]
  }
}
```

👉 Same structure:

* `dialogId`
* `meeting.meetingId`
* `meeting.transcriptSegments[]`

  * `text`
  * `speakerName` 

---

# 🧠 4. MeetingBank (HuggingFace)

## 🔹 Format: Flat JSON per instance

```json
{
  "id": "SeattleCityCouncil_12142015_CB118549",
  "transcript": "Full paragraph text...",
  "summary": "Short summary..."
}
```

👉 Key points:

* ❌ Not sentence-level (paragraphs)
* ✅ Has summary (useful later)
* ❌ No speaker info (in this version)
* ✅ Clean structured format

---

# ⚡ Final Comparison (Very Important)

| Dataset     | Structure  | Sentence Ready | Speaker Info | Cleaning Needed |
| ----------- | ---------- | -------------- | ------------ | --------------- |
| AMI TXT     | ❌ Raw text | ❌ No           | ❌ No         | 🔥 HIGH         |
| MISeD       | ✅ JSON     | ✅ Yes          | ✅ Yes        | 🟢 LOW          |
| MeetingBank | ✅ JSON     | ❌ No           | ❌ No         | 🟡 Medium       |


