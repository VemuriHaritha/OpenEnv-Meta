"""
inference.py — Robust Baseline Inference Script for Email Triage OpenEnv
===================================================================
FIXES:
- Safe OpenAI client initialization
- Works even without internet / API key
- Always falls back to rule-based policy
- No unhandled exceptions
"""

import os
import json
import time
import sys
from typing import Optional

from openai import OpenAI

# ── Load env vars ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or "dummy-key"
)

# ── Safe OpenAI client initialization ─────────────────────────────────────────
if HF_TOKEN == "dummy-key":
    print("[Info] No valid API key found. Using fallback policy only.")
    client = None
else:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"[Warning] Failed to initialize OpenAI client: {e}")
        client = None

# ── Import env ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from env import EmailTriageEnv
from models import Action

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """You are an expert email triage assistant. Your job is to analyze emails and decide how to handle them.
For each email, respond ONLY with a valid JSON object in this exact format:
{
    "category": "<spam|urgent|normal|newsletter|support|billing|hr>",
    "priority": "<low|medium|high|critical>",
    "route_to": "<inbox|trash|support|billing|hr|escalate>",
    "draft_reply": "<your reply here, or null if no reply needed>"
}
Respond ONLY with JSON. No explanation.
"""

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(obs_dict: dict, task_id: str) -> str:
    task_hints = {
        "task_easy": "Focus on correctly identifying the category and priority.",
        "task_medium": "Focus on correct routing.",
        "task_hard": "Do everything including reply drafting.",
    }

    return f"""Task: {task_hints.get(task_id, '')}

Email Details:
From: {obs_dict['sender_name']} <{obs_dict['sender']}>
Subject: {obs_dict['subject']}
Timestamp: {obs_dict['timestamp']}
Thread length: {obs_dict['thread_length']}
Has attachment: {obs_dict['has_attachment']}

Body:
{obs_dict['body']}

Triage this email now:
"""

# ── Fallback policy (ALWAYS SAFE) ─────────────────────────────────────────────
def fallback_policy(obs: dict) -> dict:
    text = (obs["subject"] + " " + obs["body"]).lower()

    if any(x in text for x in ["lottery", "viagra", "prize"]):
        return {"category": "spam", "priority": "low", "route_to": "trash", "draft_reply": None}

    if any(x in text for x in ["critical", "down", "breach", "urgent"]):
        return {"category": "urgent", "priority": "critical", "route_to": "escalate", "draft_reply": None}

    if any(x in text for x in ["invoice", "billing", "payment"]):
        return {"category": "billing", "priority": "high", "route_to": "billing", "draft_reply": None}

    if any(x in text for x in ["help", "error", "issue", "login", "crash"]):
        return {"category": "support", "priority": "high", "route_to": "support", "draft_reply": None}

    if "newsletter" in text:
        return {"category": "newsletter", "priority": "low", "route_to": "inbox", "draft_reply": None}

    return {"category": "normal", "priority": "medium", "route_to": "inbox", "draft_reply": None}

# ── LLM call (SAFE) ───────────────────────────────────────────────────────────
def call_llm(prompt: str, task_id: str, obs: dict, max_retries: int = 2) -> Optional[dict]:

    if client is None:
        return fallback_policy(obs)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            raw = response.choices[0].message.content.strip()

            # Clean markdown if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw.strip())
            return parsed

        except Exception as e:
            print(f"[Warning] LLM error attempt {attempt+1}: {e}")
            time.sleep(1)

    return fallback_policy(obs)

# ── Run task ──────────────────────────────────────────────────────────────────
def run_task(task_id: str) -> dict:
    print(f"[START] {task_id}")

    env = EmailTriageEnv(task_id=task_id, seed=42)
    obs = env.reset()

    step_results = []
    step_num = 0

    while obs is not None:
        step_num += 1
        obs_dict = obs.model_dump()

        prompt = build_user_prompt(obs_dict, task_id)
        llm_response = call_llm(prompt, task_id, obs_dict)

        try:
            action = Action(
                category=llm_response.get("category", "normal"),
                priority=llm_response.get("priority", "low"),
                route_to=llm_response.get("route_to", "inbox"),
                draft_reply=llm_response.get("draft_reply", None),
            )
        except Exception:
            action = Action(category="normal", priority="low", route_to="inbox")

        next_obs, reward, done, _ = env.step(action)

        print(f"[STEP] {task_id} #{step_num} → {action.category}, {action.priority}, {action.route_to}")

        step_results.append({
            "step": step_num,
            "email_id": obs_dict["email_id"],
            "action": action.category,
            "reward": reward.value,
        })

        obs = next_obs
        if done:
            break

    final_state = env.state()

    return {
        "task_id": task_id,
        "steps": step_num,
        "total_reward": final_state["total_reward"],
        "average_score": final_state["average_score"],
        "step_results": step_results,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[START] Running inference: {MODEL_NAME}")

    start = time.time()
    results = {}

    for task in TASKS:
        results[task] = run_task(task)

    elapsed = time.time() - start
    avg = sum(r["average_score"] for r in results.values()) / len(results)

    print(f"[END] Avg Score: {avg:.4f} | Time: {elapsed:.1f}s")

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "overall_average": avg,
            "runtime": elapsed,
        }, f, indent=2)

if __name__ == "__main__":
    main()
