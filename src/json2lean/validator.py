"""Lean 4 validator — compile files and parse compiler output.

Merges the logic from the top-level ``interact.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .models import CompileResult, Exercise, ExerciseStatus


# ------------------------------------------------------------------
# Low-level: compile a single .lean file
# ------------------------------------------------------------------

def compile_lean_file(
    filepath: str | Path,
    toolchain_dir: str | Path = "lean",
    timeout: int = 120,
) -> CompileResult:
    """Run ``lake env lean <filepath>`` and return structured output."""
    cwd = Path(toolchain_dir).resolve()
    filepath = Path(filepath).resolve()

    try:
        r = subprocess.run(
            ["lake", "env", "lean", str(filepath)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(
            filename=filepath.name,
            stdout="",
            returncode=1,
            errors=[{
                "line": 0, "column": 0,
                "message": f"Compilation timed out after {timeout}s",
                "line_content": "", "char_at_column": "",
            }],
        )

    warnings, errors = _parse_output(r.stdout, str(filepath))
    return CompileResult(
        filename=filepath.name,
        stdout=r.stdout,
        returncode=r.returncode,
        warnings=warnings,
        errors=errors,
    )


# ------------------------------------------------------------------
# Parse Lean compiler output (ported from interact.py)
# ------------------------------------------------------------------

_WARNING_RE = re.compile(r"(.+):(\d+):(\d+): warning: (.+)")
_ERROR_RE   = re.compile(r"(.+):(\d+):(\d+): error: (.+)")


def _parse_output(
    lean_output: str,
    source_path: str | None = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Parse warnings and errors from Lean compiler stdout."""
    warnings_raw: List[Dict] = []
    errors_raw: List[Dict] = []
    current: Dict | None = None
    is_warning = False

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        (warnings_raw if is_warning else errors_raw).append(current)
        current = None

    for line in lean_output.splitlines():
        wm = _WARNING_RE.match(line)
        em = _ERROR_RE.match(line)
        if wm or em:
            flush()
            fp, ln, col, msg = (wm or em).groups()  # type: ignore[union-attr]
            current = {
                "filepath": fp.strip(),
                "line": int(ln),
                "column": int(col),
                "message": msg.strip(),
                "line_content": "",
                "char_at_column": "",
            }
            is_warning = bool(wm)
        elif current is not None:
            current["message"] += "\n" + line.strip()

    flush()

    # Enrich with source-file context
    file_paths = {r["filepath"] for r in warnings_raw + errors_raw}
    contents: Dict[str, List[str] | None] = {}
    for fp in file_paths:
        try:
            contents[fp] = Path(fp).read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            contents[fp] = None

    for rec in warnings_raw + errors_raw:
        lines = contents.get(rec["filepath"])
        if lines is None:
            rec["line_content"] = f"[couldn't read file {rec['filepath']}]"
            continue
        idx = rec["line"] - 1
        if not (0 <= idx < len(lines)):
            rec["line_content"] = f"[couldn't read line {rec['line']}]"
            continue
        rec["line_content"] = lines[idx]
        if 0 <= rec["column"] < len(lines[idx]):
            rec["char_at_column"] = lines[idx][rec["column"]]

    # Strip filepath from public records (caller already knows it)
    def _clean(items: List[Dict]) -> List[Dict]:
        return [
            {k: v for k, v in r.items() if k != "filepath"}
            for r in items
        ]

    return _clean(warnings_raw), _clean(errors_raw)


# ------------------------------------------------------------------
# High-level: validate an Exercise
# ------------------------------------------------------------------

def validate_exercise(
    exercise: Exercise,
    lean_file: Path,
    toolchain_dir: str = "lean",
    timeout: int = 120,
) -> CompileResult:
    """Compile the Lean file for *exercise* and update its status."""
    result = compile_lean_file(lean_file, toolchain_dir, timeout)
    exercise.warnings = result.warnings
    exercise.errors = result.errors
    exercise.compile_returncode = result.returncode

    if exercise.is_valid:
        exercise.status = ExerciseStatus.VALID
    return result


def validate_all(
    exercises: List[Exercise],
    output_dir: Path,
    toolchain_dir: str = "lean",
    timeout: int = 120,
) -> Dict[str, CompileResult]:
    """Validate all exercises whose Lean files exist in *output_dir*."""
    results: Dict[str, CompileResult] = {}
    total = len(exercises)

    for i, ex in enumerate(exercises, 1):
        lean_file = output_dir / f"{_safe_label(ex.label)}.lean"
        if not lean_file.exists():
            print(f"[validator] [{i}/{total}] SKIP {ex.label} (no file)", file=sys.stderr)
            continue
        print(f"[validator] [{i}/{total}] compiling {ex.label}", file=sys.stderr)
        result = validate_exercise(ex, lean_file, toolchain_dir, timeout)
        results[ex.label] = result

        status = "OK" if ex.is_valid else f"FAIL ({len(result.errors)} errors)"
        print(f"[validator]   -> {status}", file=sys.stderr)

    return results


def _safe_label(label: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(label)) or "exercise"
