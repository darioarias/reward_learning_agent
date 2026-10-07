"""Scoring utilities for evaluating candidate policies."""


def score_candidate(eval_metrics: dict) -> float:
    """Return the average return used to rank a candidate policy."""
    return float(eval_metrics["avg_return"])
