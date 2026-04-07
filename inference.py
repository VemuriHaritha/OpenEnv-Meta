

import os
import json
import time
import sys
from typing import Optional

from openai import OpenAI

# ── REQUIRED ENV VARIABLES (STRICT FORMAT) ────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── INIT CLIENT ───────────────────────────────────────────────────────────────
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
except Exception as e:
    print(f"[Warning] Client init failed: {e}")
    client = None

# ── IMPORT ENV ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from env import EmailTriageEnv
from models import Action

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """Return ONLY JSON:
{
    "category": "<spam|urgent|normal|newsletter|support|billing|hr>",
    "priority": "<low|medium|high|critical>",
    "route_to": "<inbox|trash|support|billing|hr|escalate>",
    "draft_reply": "<text or null>"
}
"""

# ── PROMPT ────────────────────────────────────────────────────────────────────
def build_user_prompt(obs: dict) -> str:
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

# ── LLM CALL ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str, obs: dict, retries: int = 2) -> dict:

    for i in range(retries):
        try:
            if client is None:
                raise Exception("Client not initialized")

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

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw.strip())

        except Exception as e:
            print(f"[DEBUG] API failed attempt {i+1}: {e}")
            time.sleep(1)

    return fallback_policy(obs)

# ── TASK RUNNER ───────────────────────────────────────────────────────────────
def run_task(task_id: str):

    print("[START]")

    env = EmailTriageEnv(task_id=task_id, seed=42)
    obs = env.reset()

    step = 0

    while obs is not None:
        step += 1
        obs_dict = obs.model_dump()

        prompt = build_user_prompt(obs_dict)
        llm_out = call_llm(prompt, obs_dict)

        try:
            action = Action(
                category=llm_out.get("category", "normal"),
                priority=llm_out.get("priority", "low"),
                route_to=llm_out.get("route_to", "inbox"),
                draft_reply=llm_out.get("draft_reply", None),
            )
        except Exception:
            action = Action(category="normal", priority="low", route_to="inbox")

        obs, reward, done, _ = env.step(action)

        print("[STEP]")

        if done:
            break

    print("[END]")

    return env.state()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():

    results = {}

    for task in TASKS:
        results[task] = run_task(task)

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
