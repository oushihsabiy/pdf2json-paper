"""Preprocessor – rewrite exercise problems into structured Definition/Hypothesis/Goal form.

Merges the logic previously in ``stdjson/concise_to_lean.py``.  Can be
enabled or disabled via ``config.json`` (``preprocessing.enabled``) or
the ``--no-preprocess`` CLI flag.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from .api_client import APIClient, extract_json_value
from .loader import load_prompt
from .models import Exercise


# ------------------------------------------------------------------
# Prompt construction & validation
# ------------------------------------------------------------------

def _build_prompt(base_prompt: str, exercise_raw: Dict[str, Any], feedback: str = "") -> str:
    obj_json = json.dumps(exercise_raw, ensure_ascii=False, indent=2)
    extra = ""
    if feedback:
        extra = (
            "\n\nThe previous output was invalid. Fix it strictly according to "
            "this validation feedback:\n" + feedback + "\n"
        )
    return (
        f"{base_prompt}"
        "\n\nProcess exactly one exercise object."
        " Return exactly one JSON object and nothing else."
        " Keep every field other than \"problem\" exactly unchanged."
        " Rewrite the \"problem\" field into structured Definition / Hypothesis / Goal format"
        " suitable for Lean formalization."
        f"{extra}"
        "\n\nInput JSON object:\n"
        f"{obj_json}"
    )


def _validate_candidate(original: Dict[str, Any], candidate: Any) -> str:
    """Return an error message if *candidate* is invalid, else ``""``."""
    if isinstance(candidate, list):
        if len(candidate) != 1:
            return "Output must be a single JSON object, not an array with multiple items."
        candidate = candidate[0]

    if not isinstance(candidate, dict):
        return "Output must decode to a JSON object."

    if list(candidate.keys()) != list(original.keys()):
        return "The output object must keep exactly the same keys in exactly the same order."

    for key, value in original.items():
        if key == "problem":
            continue
        if candidate.get(key) != value:
            return f"Field '{key}' was modified, but only 'problem' may change."

    if not isinstance(candidate.get("problem"), str):
        return "Field 'problem' must remain a string."

    return ""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def preprocess_exercise(
    client: APIClient,
    exercise: Exercise,
    *,
    max_tokens: int = 4096,
    max_attempts: int = 8,
    prompt_text: str | None = None,
) -> None:
    """Rewrite ``exercise.problem`` into structured form **in-place**.

    On success, ``exercise.preprocessed_problem`` is set and
    ``exercise.raw["problem"]`` is updated.
    """
    if prompt_text is None:
        prompt_text = load_prompt("concise_to_lean")

    feedback = ""
    last_error = ""

    for _ in range(max_attempts):
        full_prompt = _build_prompt(prompt_text, exercise.raw, feedback)
        response = client.chat(
            prompt=full_prompt,
            max_tokens=max_tokens,
            call_type="preprocess",
            exercise_label=exercise.label,
            json_mode=True,
        )

        try:
            candidate = extract_json_value(response)
        except Exception as err:
            last_error = str(err)
            feedback = f"Output parsing failed: {err}"
            continue

        validation_error = _validate_candidate(exercise.raw, candidate)
        if validation_error:
            last_error = validation_error
            feedback = validation_error
            continue

        if isinstance(candidate, list):
            candidate = candidate[0]

        new_problem: str = candidate["problem"]
        exercise.preprocessed_problem = new_problem
        exercise.raw["problem"] = new_problem
        exercise.problem = new_problem
        return

    raise RuntimeError(
        f"Preprocessing failed for {exercise.label} after {max_attempts} attempts: {last_error}"
    )


def preprocess_all(
    client: APIClient,
    exercises: List[Exercise],
    *,
    max_tokens: int = 4096,
    max_attempts: int = 8,
) -> List[str]:
    """Preprocess all exercises. Returns labels that failed."""
    prompt_text = load_prompt("concise_to_lean")
    failed: List[str] = []
    total = len(exercises)

    for i, ex in enumerate(exercises, 1):
        print(f"[preprocess] [{i}/{total}] {ex.label}", file=sys.stderr)
        try:
            preprocess_exercise(
                client, ex,
                max_tokens=max_tokens,
                max_attempts=max_attempts,
                prompt_text=prompt_text,
            )
        except Exception as err:
            failed.append(ex.label)
            print(f"[preprocess] FAILED {ex.label}: {err}", file=sys.stderr)

    return failed
