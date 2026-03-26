"""Lean 4 validator — compile files and parse compiler output.

Merges the logic from the top-level ``interact.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from .models import CompileResult, Exercise, ExerciseStatus

# Number of parallel lean processes and within-process threads
_PARALLEL_JOBS = int(os.environ.get("LEAN_PARALLEL_JOBS", "4"))
_LEAN_THREADS = int(os.environ.get("LEAN_THREADS", "4"))


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
            ["lake", "env", "lean", f"--threads={_LEAN_THREADS}", str(filepath)],
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

    warnings, errors = _parse_output(r.stdout + "\n" + r.stderr, str(filepath))
    return CompileResult(
        filename=filepath.name,
        stdout=r.stdout + "\n" + r.stderr,
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
    parallel_jobs: int | None = None,
) -> Dict[str, CompileResult]:
    """Validate all exercises whose Lean files exist in *output_dir*.

    Exercises are compiled in parallel (up to *parallel_jobs* at once)
    to reduce total wall-clock time.
    """
    jobs = parallel_jobs if parallel_jobs is not None else _PARALLEL_JOBS
    results: Dict[str, CompileResult] = {}
    total = len(exercises)

    # Build the list of (exercise, lean_file) pairs that actually exist
    pending = []
    for ex in exercises:
        lean_file = output_dir / f"{_safe_label(ex.label)}.lean"
        if lean_file.exists():
            pending.append((ex, lean_file))
        else:
            print(f"[validator] SKIP {ex.label} (no file)", file=sys.stderr)

    def _compile_one(args):
        ex, lean_file = args
        return ex, lean_file, validate_exercise(ex, lean_file, toolchain_dir, timeout)

    completed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_compile_one, item): item for item in pending}
        for future in as_completed(futures):
            ex, lean_file, result = future.result()
            results[ex.label] = result
            completed += 1
            status = "OK" if ex.is_valid else f"FAIL ({len(result.errors)} errors)"
            print(
                f"[validator] [{completed}/{len(pending)}] {ex.label} -> {status}",
                file=sys.stderr,
            )

    return results


def _safe_label(label: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(label)) or "exercise"
