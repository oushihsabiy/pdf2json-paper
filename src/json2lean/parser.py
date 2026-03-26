"""Parse raw JSON data into Exercise objects."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .models import Exercise


# ------------------------------------------------------------------
# Exercise detection
# ------------------------------------------------------------------

_MARKER_KEYS = {"proof", "direct_answer", "source_idx", "source", "题目类型", "预估难度"}


def is_exercise_object(node: Any) -> bool:
    """Return True if *node* looks like an exercise dict."""
    if not isinstance(node, dict):
        return False
    if "problem" not in node:
        return False
    return any(key in node for key in _MARKER_KEYS)


def _iter_raw(node: Any) -> Iterable[Dict[str, Any]]:
    """Recursively yield exercise dicts from an arbitrarily nested structure."""
    if is_exercise_object(node):
        yield node
        return
    if isinstance(node, list):
        for item in node:
            yield from _iter_raw(item)
        return
    if isinstance(node, dict):
        for item in node.values():
            yield from _iter_raw(item)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def parse_exercises(data: Any) -> List[Exercise]:
    """Extract Exercise objects from loaded JSON data."""
    exercises: List[Exercise] = []
    for idx, raw in enumerate(_iter_raw(data), start=1):
        exercises.append(Exercise(raw=raw, index=idx))
    return exercises
