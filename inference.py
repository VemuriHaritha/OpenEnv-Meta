"""
inference.py — Baseline Inference Script for Email Triage OpenEnv
===================================================================
MANDATORY REQUIREMENTS:
- Uses OpenAI Client for all LLM calls
- Reads credentials from environment variables
- Runs all 3 tasks and produces reproducible baseline scores
- Must complete in < 20 minutes
- Works on vcpu=2, memory=8gb
RECOMMENDED BEST PRACTICES:
- Robust error handling for LLM calls and JSON parsing with retries
- Clear logging of each step, action, and reward for analysis   
- Final summary of results for all tasks
- Save results to a JSON file for later review
"""

import os
import json
import time
import sys
from typing import Optional

from openai import OpenAI

# ── Load env vars ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")

if not API_KEY:
    print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
    sys.exit(1)

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Import env directly (no HTTP server needed for inference) ─────────────────
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

Category rules:
- spam: unsolicited, promotional, or fraudulent emails
- urgent: requires immediate attention (outages, security, emergencies)
- normal: routine internal communication
- newsletter: bulk informational emails
- support: customer asking for help with a product
- billing: questions or issues about invoices and payments
- hr: human resources communications

Priority rules:
- critical: immediate action required (system down, security breach)
- high: important, handle today
- medium: handle within a few days
- low: no rush

Routing rules:
- trash: spam/irrelevant emails
- escalate: urgent/critical issues needing senior attention
- support: customer support tickets
- billing: billing department
- hr: human resources
- inbox: general emails that don't need special routing

For draft_reply:
- Write a professional, helpful reply if the email needs a response
- Use null for newsletters, spam, or routine notifications that need no reply
- Keep replies concise (2-4 sentences)

Respond ONLY with the JSON object. No explanation, no markdown, no extra text."""


def build_user_prompt(obs_dict: dict, task_id: str) -> str:
    """Build the prompt for the LLM from the observation."""
    task_hints = {
        "task_easy": "Focus on correctly identifying the category and priority.",
        "task_medium": "Focus on correct routing to the right department.",
        "task_hard": "Classify, prioritize, route, AND draft a reply if the email needs one.",
    }

    return f"""Task: {task_hints.get(task_id, '')}

Email Details:
- From: {obs_dict['sender_name']} <{obs_dict['sender']}>
- Subject: {obs_dict['subject']}
- Timestamp: {obs_dict['timestamp']}
- Thread length: {obs_dict['thread_length']} message(s)
- Has attachment: {obs_dict['has_attachment']}

Body:
{obs_dict['body']}

Triage this email now:"""


def fallback_policy(obs):
    text = (obs["subject"] + " " + obs["body"]).lower()

    if "lottery" in text or "offer" in text or "viagra" in text:
        return {"category": "spam", "priority": "low", "route_to": "trash", "draft_reply": None}

    if "critical" in text or "down" in text or "breach" in text:
        return {"category": "urgent", "priority": "critical", "route_to": "escalate", "draft_reply": None}

    if "invoice" in text or "billing" in text:
        return {"category": "billing", "priority": "high", "route_to": "billing", "draft_reply": None}

    if "login" in text or "help" in text or "error" in text or "crash" in text:
        return {"category": "support", "priority": "high", "route_to": "support", "draft_reply": None}

    return {"category": "normal", "priority": "medium", "route_to": "inbox", "draft_reply": None}


# def call_llm(prompt: str, task_id: str, max_retries: int = 3) -> Optional[dict]:
#     """Call the LLM and parse its JSON response."""
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=0.1,  # Low temp for reproducibility
#                 max_tokens=400,
#             )
#             raw = response.choices[0].message.content.strip()

#             # Strip markdown code fences if present
#             if raw.startswith("```"):
#                 raw = raw.split("```")[1]
#                 if raw.startswith("json"):
#                     raw = raw[4:]
#             raw = raw.strip()

#             parsed = json.loads(raw)
#             return parsed

#         except json.JSONDecodeError as e:
#             print(f"  [Warning] JSON parse error (attempt {attempt+1}): {e}")
#             if attempt == max_retries - 1:
#                 # Return safe fallback
#                 return {
#                     "category": "normal",
#                     "priority": "low",
#                     "route_to": "inbox",
#                     "draft_reply": None,
#                 }
#         except Exception as e:
#             print(f"  [Warning] LLM call error (attempt {attempt+1}): {e}")
#             time.sleep(2)
#             if attempt == max_retries - 1:
#                 return None

#     return None

def call_llm(prompt: str, task_id: str, obs: dict, max_retries: int = 3) -> Optional[dict]:
    """Call the LLM and parse its JSON response with robust fallback."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=400,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            return parsed

        except json.JSONDecodeError as e:
            print(f"  [Warning] JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(1)

        except Exception as e:
            print(f"  [Warning] LLM call error (attempt {attempt+1}): {e}")
            time.sleep(2)

    # Final fallback after all retries fail
    print("  [Fallback] Using rule-based policy")
    return fallback_policy(obs)


def run_task(task_id: str) -> dict:
    """Run a single task and return results."""
    print(f"\n{'='*60}")
    print(f"Running Task: {task_id.upper()}")
    print(f"{'='*60}")

    env = EmailTriageEnv(task_id=task_id, seed=42)
    obs = env.reset()

    step_results = []
    step_num = 0

    while obs is not None:
        step_num += 1
        obs_dict = obs.model_dump()
        print(f"\nStep {step_num}: Processing email '{obs_dict['subject'][:50]}...'")

        # Build prompt and call LLM
        prompt = build_user_prompt(obs_dict, task_id)
        llm_response = call_llm(prompt, task_id,obs_dict)

        if llm_response is None:
            print("  [Error] LLM failed, using fallback action")
            llm_response = {"category": "normal", "priority": "low", "route_to": "inbox", "draft_reply": None}

        # Validate and create action
        try:
            action = Action(
                category=llm_response.get("category", "normal"),
                priority=llm_response.get("priority", "low"),
                route_to=llm_response.get("route_to", "inbox"),
                draft_reply=llm_response.get("draft_reply", None),
            )
        except Exception as e:
            print(f"  [Warning] Action validation failed: {e}. Using fallback.")
            action = Action(category="normal", priority="low", route_to="inbox")

        print(f"  → category={action.category}, priority={action.priority}, route={action.route_to}")

        # Take step
        next_obs, reward, done, info = env.step(action)

        print(f"  → reward={reward.value:.3f} (cat={reward.category_score:.1f}, pri={reward.priority_score:.1f}, route={reward.routing_score:.1f}, reply={reward.reply_score:.1f})")

        step_results.append({
            "step": step_num,
            "email_id": obs_dict["email_id"],
            "subject": obs_dict["subject"][:60],
            "action": {
                "category": action.category,
                "priority": action.priority,
                "route_to": action.route_to,
                "has_reply": bool(action.draft_reply),
            },
            "reward": reward.value,
            "breakdown": reward.breakdown,
        })

        obs = next_obs
        if done:
            break

    # Final state
    final_state = env.state()
    avg_score = final_state["average_score"]
    total_reward = final_state["total_reward"]

    print(f"\n{'─'*40}")
    print(f"Task '{task_id}' complete!")
    print(f"  Steps: {step_num}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Average Score: {avg_score:.4f}")
    print(f"{'─'*40}")

    return {
        "task_id": task_id,
        "steps": step_num,
        "total_reward": total_reward,
        "average_score": avg_score,
        "step_results": step_results,
    }


def main():
    print("="*60)
    print("Email Triage OpenEnv — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print("="*60)

    start_time = time.time()
    all_results = {}

    for task_id in TASKS:
        result = run_task(task_id)
        all_results[task_id] = result

    # Print final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'Avg Score':>12} {'Total Reward':>14} {'Steps':>8}")
    print(f"{'─'*52}")
    for task_id, res in all_results.items():
        print(f"{task_id:<15} {res['average_score']:>12.4f} {res['total_reward']:>14.4f} {res['steps']:>8}")

    overall_avg = sum(r["average_score"] for r in all_results.values()) / len(all_results)
    print(f"{'─'*52}")
    print(f"{'OVERALL AVG':<15} {overall_avg:>12.4f}")
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Save results to file
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "results": all_results,
            "overall_average": overall_avg,
            "runtime_seconds": elapsed,
        }, f, indent=2)
    print("Results saved to baseline_results.json")

    return all_results


if __name__ == "__main__":
    main()
