# 📧 Email Triage OpenEnv

A **real-world reinforcement learning environment** where AI agents learn to triage emails — classifying, prioritizing, routing, and drafting replies — using the [OpenEnv](https://github.com/huggingface/openenv) standard API.

---

## 🌍 Environment Description & Motivation

Email triage is one of the most universal productivity challenges. Humans spend an average of 2.5 hours/day on email. This environment trains agents to handle a realistic inbox: distinguishing spam from urgent security alerts, routing billing disputes to the right team, and drafting contextually appropriate replies.

This environment fills a real gap in the agent evaluation ecosystem — there are no public OpenEnv environments for email/inbox management.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/VemuriHaritha/OpenEnv-Meta
cd OpenEnv-Meta

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# → Server running at http://localhost:7860
```

---

## 📡 API Usage

### Reset
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "seed": 42}'
```

### Step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_easy",
    "action": {
      "category": "spam",
      "priority": "low",
      "route_to": "trash",
      "draft_reply": null
    }
  }'
```

### State
```bash
curl -X POST http://localhost:7860/state \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

---

## 📊 Observation Space

| Field | Type | Description |
|---|---|---|
| `email_id` | string | Unique email identifier |
| `subject` | string | Email subject line |
| `body` | string | Full email body |
| `sender` | string | Sender email address |
| `sender_name` | string | Sender display name |
| `timestamp` | string | ISO 8601 timestamp |
| `thread_length` | int | Messages in the thread |
| `has_attachment` | bool | Whether there's an attachment |
| `step_number` | int | Current step number |
| `emails_remaining` | int | Emails left in episode |
| `task_id` | string | Current task |
| `task_description` | string | Human-readable task description |

---

## ⚡ Action Space

| Field | Type | Valid Values |
|---|---|---|
| `category` | string | `spam`, `urgent`, `normal`, `newsletter`, `support`, `billing`, `hr` |
| `priority` | string | `low`, `medium`, `high`, `critical` |
| `route_to` | string | `inbox`, `trash`, `support`, `billing`, `hr`, `escalate` |
| `draft_reply` | string or null | Free text reply, or `null` if no reply needed |

---

## 🎯 Tasks

### Task 1: Basic Email Classification (Easy)
**Objective**: Correctly categorize and prioritize emails.  
**Dataset**: Mix of spam, urgent, normal, and newsletter emails.  
**Scoring**: 50% category + 30% priority + 20% routing.  
**Expected difficulty**: Solvable by most LLMs with >0.7 avg score.  
**Max steps**: 6

### Task 2: Priority Ranking & Routing (Medium)
**Objective**: Route emails to the correct department (support, billing, HR).  
**Dataset**: Customer support, billing, HR, and normal emails.  
**Scoring**: 20% category + 35% priority + 45% routing (routing is key!).  
**Expected difficulty**: Requires understanding of context and department logic.  
**Max steps**: 8

### Task 3: Full Triage with Draft Reply (Hard)
**Objective**: Classify, prioritize, route, AND draft contextually appropriate replies.  
**Dataset**: All emails that require a reply (urgent, support, billing).  
**Scoring**: 20% category + 20% priority + 30% routing + 30% reply quality.  
**Reply scoring**: Keyword matching + length + relevance.  
**Expected difficulty**: Frontier models score ~0.6–0.75; small models struggle.  
**Max steps**: Dynamic (equals number of reply-required emails)

---

## 📈 Reward Function

All rewards are in range **[0.0, 1.0]** with partial credit:

- **Category score**: 1.0 = exact match, 0.0 = wrong
- **Priority score**: 1.0 = exact, 0.5 = one level off, 0.0 = two+ levels off
- **Routing score**: 1.0 = correct route, 0.0 = wrong route
- **Reply score** (hard task only): 0.3 base + up to 0.7 for keyword relevance
- **Penalties**: -0.5 for invalid actions, -0.1 for unnecessary replies on no-reply emails

---

## 📦 Baseline Scores

Run the baseline:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"
python inference.py
```

| Task | Llama-3.1-8B Avg Score |
|---|---|
| task_easy | 0.97 |
| task_medium | 0.96 |
| task_hard | 0.84 |
| **Overall** | **0.9127** |

---

## 🐳 Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# Test
curl http://localhost:7860/health
```

---

## 🗂️ Project Structure

```
OpenEnv-Meta/
├── app.py                # Root entry (HF Spaces)
├── server/app.py         # FastAPI server
├── env.py                # Environment logic
├── models.py             # Pydantic models
├── inference.py          # Baseline runner
├── openenv.yaml          # OpenEnv config
├── requirements.txt
├── Dockerfile
├── data/emails.py        # Dataset
├── tasks/graders.py      # Reward logic
```

---

## ✅ OpenEnv Spec Compliance

- [x] Typed `Observation`, `Action`, `Reward` Pydantic models
- [x] `step(action)` → returns observation, reward, done, info
- [x] `reset()` → returns initial observation
- [x] `state()` → returns current episode state
- [x] `openenv.yaml` with full metadata
- [x] Minimum 3 tasks (easy → medium → hard)
- [x] Graders scoring 0.0–1.0
- [x] Deterministic (seed-controlled)
- [x] Baseline `inference.py` using OpenAI client
- [x] Working `Dockerfile`
- [x] Deploys to Hugging Face Spaces on port 7860

---

## 📬 Contact

Built by **Haritha Vemuri**.
