"""
env.py — Core Email Triage OpenEnv Environment
Implements full OpenEnv spec: step() / reset() / state()
"""

import copy
import random
import sys
import os
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))

from models import Observation, Action, Reward, EpisodeState
from data.emails import EMAILS, TASK_EASY_EMAILS, TASK_MEDIUM_EMAILS, TASK_HARD_EMAILS
from tasks.graders import GRADERS

TASK_CONFIG = {
    "task_easy": {
        "emails": TASK_EASY_EMAILS,
        "description": "Classify emails into basic categories (spam/urgent/normal/newsletter) and assign priority.",
        "max_steps": 6,
    },
    "task_medium": {
        "emails": TASK_MEDIUM_EMAILS,
        "description": "Classify, prioritize, and route emails to the correct department (support/billing/hr/inbox).",
        "max_steps": 8,
    },
    "task_hard": {
        "emails": TASK_HARD_EMAILS,
        "description": "Fully triage emails: classify, prioritize, route, AND draft an appropriate reply for emails that need one.",
        "max_steps": len([e for e in EMAILS if e["ground_truth"]["reply_needed"]]),
    },
}


class EmailTriageEnv:
    """
    Real-world email triage environment.
    An agent must classify, prioritize, route, and (in hard mode) reply to emails.
    
    Usage:
        env = EmailTriageEnv(task_id="task_easy")
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    def __init__(self, task_id: str = "task_easy", seed: Optional[int] = 42):
        assert task_id in TASK_CONFIG, f"Invalid task_id. Choose from: {list(TASK_CONFIG.keys())}"
        self.task_id = task_id
        self.seed = seed
        self._config = TASK_CONFIG[task_id]
        self._emails = copy.deepcopy(self._config["emails"])
        self._max_steps = self._config["max_steps"]
        self._episode_state: Optional[EpisodeState] = None
        self._current_emails: list = []
        self._rng = random.Random(seed)

    def reset(self) -> Observation:
        """Reset environment to initial state, return first observation."""
        self._rng = random.Random(self.seed)
        self._current_emails = copy.deepcopy(self._emails)
        self._rng.shuffle(self._current_emails)
        self._current_emails = self._current_emails[:self._max_steps]

        self._episode_state = EpisodeState(
            task_id=self.task_id,
            step=0,
            max_steps=self._max_steps,
            done=False,
            total_reward=0.0,
            emails_processed=0,
            current_email_index=0,
            scores=[],
        )

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        """
        Process one agent action.
        Returns: (next_observation, reward, done, info)
        """
        if self._episode_state is None:
            raise RuntimeError("Call reset() before step()")
        if self._episode_state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Validate action fields
        try:
            action.validate_fields()
        except AssertionError as e:
            # Penalize invalid actions
            reward = Reward(
                value=-0.5,
                category_score=0.0,
                priority_score=0.0,
                routing_score=0.0,
                reply_score=0.0,
                breakdown={"error": str(e), "penalty": "invalid_action"}
            )
            self._episode_state.step += 1
            self._episode_state.scores.append(-0.5)
            done = self._check_done()
            obs = self._make_observation() if not done else None
            return obs, reward, done, {"error": str(e)}

        # Grade the action
        current_email = self._current_emails[self._episode_state.current_email_index]
        ground_truth = current_email["ground_truth"]
        grader = GRADERS[self.task_id]
        reward = grader(action, ground_truth)

        # Update state
        self._episode_state.step += 1
        self._episode_state.emails_processed += 1
        self._episode_state.current_email_index += 1
        self._episode_state.total_reward += reward.value
        self._episode_state.scores.append(reward.value)

        # Check if episode is done
        done = self._check_done()
        self._episode_state.done = done

        # Build next observation
        next_obs = self._make_observation() if not done else None

        info = {
            "step": self._episode_state.step,
            "email_id": current_email["email_id"],
            "reward_breakdown": reward.breakdown,
            "total_reward": self._episode_state.total_reward,
            "emails_remaining": len(self._current_emails) - self._episode_state.current_email_index,
        }

        return next_obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return the current episode state as a dictionary."""
        if self._episode_state is None:
            return {"status": "not_started", "task_id": self.task_id}

        avg_score = (
            sum(self._episode_state.scores) / len(self._episode_state.scores)
            if self._episode_state.scores else 0.0
        )

        return {
            "task_id": self._episode_state.task_id,
            "step": self._episode_state.step,
            "max_steps": self._episode_state.max_steps,
            "done": self._episode_state.done,
            "total_reward": round(self._episode_state.total_reward, 4),
            "average_score": round(avg_score, 4),
            "emails_processed": self._episode_state.emails_processed,
            "emails_remaining": max(
                0, len(self._current_emails) - self._episode_state.current_email_index
            ),
            "scores_per_step": self._episode_state.scores,
            "task_description": self._config["description"],
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _make_observation(self) -> Optional[Observation]:
        """Build observation from the current email."""
        idx = self._episode_state.current_email_index
        if idx >= len(self._current_emails):
            return None

        email = self._current_emails[idx]
        remaining = len(self._current_emails) - idx - 1

        return Observation(
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            sender_name=email["sender_name"],
            timestamp=email["timestamp"],
            thread_length=email["thread_length"],
            has_attachment=email["has_attachment"],
            step_number=self._episode_state.step,
            emails_remaining=remaining,
            task_id=self.task_id,
            task_description=self._config["description"],
        )

    def _check_done(self) -> bool:
        """Episode ends when all emails processed or max steps reached."""
        return (
            self._episode_state.current_email_index >= len(self._current_emails)
            or self._episode_state.step >= self._max_steps
        )