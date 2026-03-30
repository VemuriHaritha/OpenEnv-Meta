"""
tasks/graders.py — Deterministic graders for all 3 tasks.
Each grader scores an agent action against ground truth (0.0 to 1.0).
"""

from typing import Tuple
from models import Action, Reward


def grade_easy(action: Action, ground_truth: dict) -> Reward:
    """
    TASK 1 (Easy): Basic Email Classification
    Focus: category + priority correctness.
    Score range: 0.0 – 1.0
    """
    category_score = 1.0 if action.category == ground_truth["category"] else 0.0

    # Priority: exact match = 1.0, one level off = 0.5, more = 0.0
    priority_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    pred_p = priority_order.get(action.priority, -1)
    true_p = priority_order.get(ground_truth["priority"], -1)
    diff = abs(pred_p - true_p)
    priority_score = 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)

    routing_score = 1.0 if action.route_to == ground_truth["route_to"] else 0.0

    # Weighted: category most important in easy task
    value = 0.5 * category_score + 0.3 * priority_score + 0.2 * routing_score

    # Penalty for unnecessary draft reply in easy task
    if action.draft_reply and len(action.draft_reply) > 10:
        value = max(0.0, value - 0.05)

    return Reward(
        value=round(value, 3),
        category_score=category_score,
        priority_score=priority_score,
        routing_score=routing_score,
        reply_score=0.0,
        breakdown={
            "task": "easy",
            "category_correct": action.category == ground_truth["category"],
            "priority_diff": diff,
            "routing_correct": action.route_to == ground_truth["route_to"],
        }
    )


def grade_medium(action: Action, ground_truth: dict) -> Reward:
    """
    TASK 2 (Medium): Priority Ranking & Routing
    Focus: routing correctness + priority ordering.
    Score range: 0.0 – 1.0
    """
    category_score = 1.0 if action.category == ground_truth["category"] else 0.3

    priority_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    pred_p = priority_order.get(action.priority, -1)
    true_p = priority_order.get(ground_truth["priority"], -1)
    diff = abs(pred_p - true_p)
    priority_score = 1.0 if diff == 0 else (0.6 if diff == 1 else 0.2)

    routing_score = 1.0 if action.route_to == ground_truth["route_to"] else 0.0

    # In medium task, routing is weighted highest
    value = 0.2 * category_score + 0.35 * priority_score + 0.45 * routing_score

    return Reward(
        value=round(value, 3),
        category_score=category_score,
        priority_score=priority_score,
        routing_score=routing_score,
        reply_score=0.0,
        breakdown={
            "task": "medium",
            "category_score": category_score,
            "priority_diff": diff,
            "routing_correct": action.route_to == ground_truth["route_to"],
        }
    )


def grade_hard(action: Action, ground_truth: dict) -> Reward:
    """
    TASK 3 (Hard): Full Triage with Draft Reply
    Focus: all dimensions + quality of drafted reply.
    Score range: 0.0 – 1.0
    """
    category_score = 1.0 if action.category == ground_truth["category"] else 0.2

    priority_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    pred_p = priority_order.get(action.priority, -1)
    true_p = priority_order.get(ground_truth["priority"], -1)
    diff = abs(pred_p - true_p)
    priority_score = 1.0 if diff == 0 else (0.5 if diff == 1 else 0.1)

    routing_score = 1.0 if action.route_to == ground_truth["route_to"] else 0.0

    # Draft reply grading
    reply_score = 0.0
    reply_needed = ground_truth.get("reply_needed", False)
    ideal_keywords = ground_truth.get("ideal_reply_keywords", [])

    if reply_needed:
        if not action.draft_reply or len(action.draft_reply.strip()) < 20:
            reply_score = 0.0  # No reply when one was needed
        else:
            reply_text = action.draft_reply.lower()
            # Base score for having a reply
            reply_score = 0.3
            # Keyword matching (partial credit)
            if ideal_keywords:
                matched = sum(1 for kw in ideal_keywords if kw.lower() in reply_text)
                reply_score += 0.7 * (matched / len(ideal_keywords))
            else:
                reply_score = 0.6
            # Penalize very short replies
            if len(action.draft_reply) < 50:
                reply_score *= 0.5
    else:
        # No reply needed — penalize if they wrote one anyway
        if action.draft_reply and len(action.draft_reply.strip()) > 10:
            reply_score = -0.1
        else:
            reply_score = 1.0

    reply_score = round(max(0.0, min(1.0, reply_score)), 3)

    # Hard task: all dimensions matter equally, reply is 30%
    value = (
        0.20 * category_score
        + 0.20 * priority_score
        + 0.30 * routing_score
        + 0.30 * reply_score
    )

    # Penalty for invalid actions on hard task
    if action.category not in {"spam", "urgent", "normal", "newsletter", "support", "billing", "hr"}:
        value *= 0.5

    return Reward(
        value=round(max(0.0, min(1.0, value)), 3),
        category_score=category_score,
        priority_score=priority_score,
        routing_score=routing_score,
        reply_score=reply_score,
        breakdown={
            "task": "hard",
            "category_score": category_score,
            "priority_diff": diff,
            "routing_correct": action.route_to == ground_truth["route_to"],
            "reply_needed": reply_needed,
            "reply_length": len(action.draft_reply or ""),
            "keywords_matched": sum(
                1 for kw in ideal_keywords
                if kw.lower() in (action.draft_reply or "").lower()
            ) if ideal_keywords else "N/A"
        }
    )


GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}