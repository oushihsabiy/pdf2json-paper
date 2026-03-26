"""Automatic Lean code recovery via LLM.

When validation fails, this module sends the broken Lean code together with
the compiler errors to the LLM and asks it to produce a corrected version.
The validate → recover cycle repeats up to a configurable limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

from .api_client import APIClient, extract_lean_code
from .loader import load_prompt
from .models import Exercise, ExerciseStatus
from .validator import validate_exercise
from .writer import write_lean_file


def _format_errors(exercise: Exercise) -> str:
    """Format compiler errors into a readable block for the LLM."""
    parts: List[str] = []
    for err in exercise.errors:
        loc = f"line {err.get('line', '?')}:{err.get('column', '?')}"
        parts.append(f"  {loc}: {err.get('message', '')}")
        lc = err.get("line_content", "")
        if lc:
            parts.append(f"    | {lc}")
    return "\n".join(parts)


def _build_prompt(base_prompt: str, lean_code: str, error_text: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"== Lean code ==\n```lean\n{lean_code}\n```\n\n"
        f"== Compiler errors ==\n{error_text}\n\n"
        "Output ONLY the corrected Lean file. No explanations."
    )


def recover_exercise(
    client: APIClient,
    exercise: Exercise,
    output_dir: Path,
    *,
    toolchain_dir: str = "lean",
    lean_timeout: int = 120,
    max_tokens: int = 4096,
    max_retries: int = 5,
    prompt_text: str | None = None,
) -> bool:
    """Attempt to fix *exercise* through repeated validate→recover cycles.

    Returns ``True`` if the exercise eventually compiles, ``False`` otherwise.
    """
    if prompt_text is None:
        prompt_text = load_prompt("recovery")

    for attempt in range(1, max_retries + 1):
        if exercise.is_valid:
            return True

        error_text = _format_errors(exercise)
        if not error_text:
            error_text = "(no structured errors; returncode was non-zero)"

        print(
            f"[recover] {exercise.label} attempt {attempt}/{max_retries}",
            file=sys.stderr,
        )

        full_prompt = _build_prompt(prompt_text, exercise.lean_code, error_text)
        response = client.chat(
            prompt=full_prompt,
            max_tokens=max_tokens,
            call_type="recover",
            exercise_label=exercise.label,
        )

        try:
            new_code = extract_lean_code(response)
        except Exception as err:
            print(f"[recover]   extraction failed: {err}", file=sys.stderr)
            continue

        if not new_code.strip():
            print("[recover]   empty code returned, skipping", file=sys.stderr)
            continue

        exercise.lean_code = new_code
        exercise.repair_attempts = attempt

        # Re-write and re-validate
        lean_file = write_lean_file(exercise, output_dir)
        validate_exercise(exercise, lean_file, toolchain_dir, lean_timeout)

        if exercise.is_valid:
            print(f"[recover]   FIXED on attempt {attempt}", file=sys.stderr)
            exercise.status = ExerciseStatus.VALID
            return True

        print(
            f"[recover]   still {len(exercise.errors)} error(s) after attempt {attempt}",
            file=sys.stderr,
        )

    exercise.status = ExerciseStatus.REPAIR_FAILED
    return False


def recover_all(
    client: APIClient,
    exercises: List[Exercise],
    output_dir: Path,
    *,
    toolchain_dir: str = "lean",
    lean_timeout: int = 120,
    max_tokens: int = 4096,
    max_retries: int = 5,
) -> List[str]:
    """Attempt recovery for every exercise that failed validation.

    Returns labels that could not be fixed.
    """
    prompt_text = load_prompt("recovery")
    still_broken: List[str] = []

    for ex in exercises:
        if ex.is_valid or ex.status == ExerciseStatus.ERROR:
            continue

        ok = recover_exercise(
            client, ex, output_dir,
            toolchain_dir=toolchain_dir,
            lean_timeout=lean_timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            prompt_text=prompt_text,
        )
        if not ok:
            still_broken.append(ex.label)

    return still_broken
