"""
app.py — FastAPI server exposing OpenEnv step()/reset()/state() API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import Observation, Action, Reward
from env import EmailTriageEnv

app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for AI agent training and evaluation.",
    version="1.0.0",
)

# One environment instance per task (simple, works for single-user HF Space)
_envs: Dict[str, EmailTriageEnv] = {}


def _get_env(task_id: str = "task_easy") -> EmailTriageEnv:
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id)
    return _envs[task_id]


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ── OpenEnv API ───────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    seed: Optional[int] = 42


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """Reset the environment and return the first observation."""

    # Default values if no body is provided
    task_id = req.task_id if req else "task_easy"
    seed = req.seed if req else 42

    valid_tasks = ["task_easy", "task_medium", "task_hard"]
    if task_id not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task_id. Choose from: {valid_tasks}")

    env = EmailTriageEnv(task_id=task_id, seed=seed)
    _envs[task_id] = env

    obs = env.reset()
    return obs.model_dump()


class StepRequest(BaseModel):
    task_id: str = "task_easy"
    action: Action


@app.post("/step")
def step(req: StepRequest):
    """Take one step in the environment."""
    env = _get_env(req.task_id)
    if env._episode_state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump() if obs else None,
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


class StateRequest(BaseModel):
    task_id: str = "task_easy"


@app.post("/state")
def state(req: StateRequest):
    """Return the current environment state."""
    env = _get_env(req.task_id)
    return env.state()


@app.get("/state/{task_id}")
def state_get(task_id: str):
    """Return the current environment state (GET version)."""
    env = _get_env(task_id)
    return env.state()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
