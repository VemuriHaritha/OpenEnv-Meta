import os
import json
import time
import sys
from typing import List, Optional

from openai import OpenAI

# ── ENV VARIABLES (STRICT) ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("API_KEY")

# ── INIT CLIENT ───────────────────────────────────────────────────────────────
if HF_TOKEN:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"[DEBUG] client_init_error={e}", flush=True)
        client = None
else:
    print("[DEBUG] No API key found", flush=True)
    client = None

# ── IMPORT ENV ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from env import EmailTriageEnv
from models import Action

TASKS = ["task_easy", "task_medium", "task_hard"]

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Return ONLY JSON:
{
    "category": "<spam|urgent|normal|newsletter|support|billing|hr>",
    "priority": "<low|medium|high|critical>",
    "route_to": "<inbox|trash|support|billing|hr|escalate>",
    "draft_reply": "<text or null>"
}
"""

# ── LOGGING (STRICT FORMAT) ───────────────────────────────────────────────────
def log_start(task: str):
    print(f"[START] task={task} env=email_triage model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── PROMPT ────────────────────────────────────────────────────────────────────
def build_prompt(obs: dict) -> str:
    return f"""
From: {obs['sender_name']} <{obs['sender']}>
Subject: {obs['subject']}
Body:
{obs['body']}
"""

# ── FALLBACK ──────────────────────────────────────────────────────────────────
def fallback_policy(obs: dict) -> dict:
    text = (obs["subject"] + " " + obs["body"]).lower()

    if "lottery" in text:
        return {"category": "spam", "priority": "low", "route_to": "trash", "draft_reply": None}
    if "urgent" in text:
        return {"category": "urgent", "priority": "critical", "route_to": "escalate", "draft_reply": None}
    if "invoice" in text:
        return {"category": "billing", "priority": "high", "route_to": "billing", "draft_reply": None}
    if "help" in text:
        return {"category": "support", "priority": "high", "route_to": "support", "draft_reply": None}

    return {"category": "normal", "priority": "medium", "route_to": "inbox", "draft_reply": None}

# ── LLM CALL (ENSURES API USAGE) ──────────────────────────────────────────────
def call_llm(prompt: str, obs: dict, retries: int = 2) -> dict:

    for i in range(retries):
        try:
            if client is None:
                raise Exception("client not available")

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            raw = (response.choices[0].message.content or "").strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw.strip())
        except Exception as e:
            print(f"[DEBUG] api_error={e}", flush=True)
            time.sleep(1)

    return fallback_policy(obs)

# ── RUN TASK ──────────────────────────────────────────────────────────────────
def run_task(task_id: str):

    log_start(task_id)

    env = EmailTriageEnv(task_id=task_id, seed=42)
    obs = env.reset()

    rewards = []
    steps = 0

    try:
        while obs is not None:
            steps += 1
            obs_dict = obs.model_dump()

            prompt = build_prompt(obs_dict)
            llm_out = call_llm(prompt, obs_dict)

            try:
                action = Action(
                    category=llm_out.get("category", "normal"),
                    priority=llm_out.get("priority", "low"),
                    route_to=llm_out.get("route_to", "inbox"),
                    draft_reply=llm_out.get("draft_reply", None),
                )
                action_str = action.category
            except Exception as e:
                action = Action(category="normal", priority="low", route_to="inbox")
                action_str = "fallback"

            obs, reward, done, _ = env.step(action)

            r = reward.value if reward else 0.0
            rewards.append(r)

            log_step(steps, action_str, r, done, None)

            if done:
                break

    except Exception as e:
        print(f"[DEBUG] runtime_error={e}", flush=True)

    finally:
        try:
            env.close()
        except:
            pass

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)

        success = score > 0.3

        log_end(success, steps, score, rewards)

    return score

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():

    results = {}
    for task in TASKS:
        results[task] = run_task(task)

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
