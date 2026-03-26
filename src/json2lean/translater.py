"""Translate a preprocessed exercise into Lean 4 code via the LLM."""

from __future__ import annotations

import json
import sys
from typing import List

from .api_client import APIClient, extract_lean_code
from .comment_builder import build_comment
from .loader import load_prompt
from .models import Exercise, ExerciseStatus


def _build_prompt(base_prompt: str, exercise: Exercise) -> str:
    obj_json = json.dumps(exercise.raw, ensure_ascii=False, indent=2)
    return f"{base_prompt}\n\n{obj_json}"


def translate_exercise(
    client: APIClient,
    exercise: Exercise,
    *,
    max_tokens: int = 4096,
    max_attempts: int = 3,
    prompt_text: str | None = None,
) -> None:
    """Translate one exercise to Lean 4 code **in-place**.

    Sets ``exercise.lean_code`` and updates ``exercise.status``.
    """
    if prompt_text is None:
        prompt_text = load_prompt("json_to_lean")

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        full_prompt = _build_prompt(prompt_text, exercise)
        response = client.chat(
            prompt=full_prompt,
            max_tokens=max_tokens,
            call_type="translate",
            exercise_label=exercise.label,
        )
        try:
            code = extract_lean_code(response)
        except Exception as err:
            last_error = str(err)
            print(
                f"[translate] attempt {attempt} failed for {exercise.label}: {err}",
                file=sys.stderr,
            )
            continue

        if not code.strip():
            last_error = "Empty Lean code returned"
            continue

        exercise.lean_code = code
        exercise.status = ExerciseStatus.TRANSLATED
        return

    raise RuntimeError(
        f"Translation failed for {exercise.label} after {max_attempts} attempts: {last_error}"
    )


def translate_all(
    client: APIClient,
    exercises: List[Exercise],
    *,
    max_tokens: int = 4096,
    max_attempts: int = 3,
) -> List[str]:
    """Translate every exercise. Returns labels that failed."""
    prompt_text = load_prompt("json_to_lean")
    failed: List[str] = []
    total = len(exercises)

    for i, ex in enumerate(exercises, 1):
        print(f"[translate] [{i}/{total}] {ex.label}", file=sys.stderr)
        try:
            translate_exercise(
                client, ex,
                max_tokens=max_tokens,
                max_attempts=max_attempts,
                prompt_text=prompt_text,
            )
        except Exception as err:
            failed.append(ex.label)
            ex.status = ExerciseStatus.ERROR
            print(f"[translate] FAILED {ex.label}: {err}", file=sys.stderr)

    return failed
