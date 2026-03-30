"""
models.py — Typed Pydantic models for Email Triage OpenEnv
Defines Observation, Action, Reward as required by OpenEnv spec.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent sees at each step."""
    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    sender: str = Field(..., description="Sender email address")
    sender_name: str = Field(..., description="Sender display name")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the email")
    thread_length: int = Field(..., description="Number of messages in the thread")
    has_attachment: bool = Field(default=False, description="Whether email has attachments")
    step_number: int = Field(..., description="Current step in the episode")
    emails_remaining: int = Field(..., description="Emails left to triage in this episode")
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Human-readable description of the task")


class Action(BaseModel):
    """What the agent does in response to an email."""
    category: str = Field(
        ...,
        description="Email category: spam | urgent | normal | newsletter | support | billing | hr"
    )
    priority: str = Field(
        ...,
        description="Priority level: low | medium | high | critical"
    )
    route_to: str = Field(
        ...,
        description="Routing destination: inbox | trash | support | billing | hr | escalate"
    )
    draft_reply: Optional[str] = Field(
        default=None,
        description="Optional draft reply text (required for hard task)"
    )

    def validate_fields(self):
        valid_categories = {"spam", "urgent", "normal", "newsletter", "support", "billing", "hr"}
        valid_priorities = {"low", "medium", "high", "critical"}
        valid_routes = {"inbox", "trash", "support", "billing", "hr", "escalate"}
        assert self.category in valid_categories, f"Invalid category: {self.category}"
        assert self.priority in valid_priorities, f"Invalid priority: {self.priority}"
        assert self.route_to in valid_routes, f"Invalid route: {self.route_to}"


class Reward(BaseModel):
    """Reward signal returned after each action."""
    value: float = Field(..., description="Reward value in range [-1.0, 1.0]")
    category_score: float = Field(..., description="Score for category classification [0,1]")
    priority_score: float = Field(..., description="Score for priority assignment [0,1]")
    routing_score: float = Field(..., description="Score for routing decision [0,1]")
    reply_score: float = Field(default=0.0, description="Score for draft reply quality [0,1]")
    breakdown: dict = Field(default_factory=dict, description="Human-readable score breakdown")


class EpisodeState(BaseModel):
    """Full internal state of the environment."""
    task_id: str
    step: int
    max_steps: int
    done: bool
    total_reward: float
    emails_processed: int
    current_email_index: int
    scores: List[float] = Field(default_factory=list)