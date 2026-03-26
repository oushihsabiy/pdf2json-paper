"""Write Lean files to disk."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

from .models import Exercise


def _safe_filename(label: str) -> str:
    """Turn an exercise label into a safe filename stem."""
    safe = re.sub(r"[^\w\-.]", "_", str(label))
    return safe or "exercise"


def write_lean_file(exercise: Exercise, output_dir: Path) -> Path:
    """Write one exercise's Lean code to *output_dir*/<label>.lean."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(exercise.label)
    path = output_dir / f"{stem}.lean"
    path.write_text(exercise.lean_code + "\n", encoding="utf-8")
    return path


def write_all(exercises: List[Exercise], output_dir: Path) -> List[Path]:
    """Write all translated exercises to disk. Returns list of written paths."""
    paths: List[Path] = []
    for ex in exercises:
        if not ex.lean_code:
            continue
        p = write_lean_file(ex, output_dir)
        paths.append(p)
        print(f"[writer] wrote {p}", file=sys.stderr)
    return paths
