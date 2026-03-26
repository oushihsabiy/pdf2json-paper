"""Build Lean block-comment metadata from exercise fields."""

from __future__ import annotations

from typing import Any, Dict


_COMMENT_FIELDS = [
    "index",
    "source_idx",
    "source",
    "题目类型",
    "预估难度",
    "problem",
    "proof",
    "direct_answer",
]


def build_comment(raw: Dict[str, Any]) -> str:
    """Return a Lean block comment containing all metadata fields."""
    lines = ["/-"]
    for key in _COMMENT_FIELDS:
        val = raw.get(key, "")
        lines.append(f"{key}: {val}")
    lines.append("-/")
    return "\n".join(lines)
