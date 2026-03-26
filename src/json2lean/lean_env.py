"""Lean 4 environment detection and helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


LEAN_SETUP_INSTRUCTIONS = """\
Lean 4 is required but was not detected.  Install it with elan:

  1.  Install elan (the Lean version manager):
        curl https://elan-init.trycloudflare.com/elan/elan-init.sh -sSf | sh

      Or via Homebrew on macOS:
        brew install elan-init

  2.  Install a Lean 4 toolchain:
        elan default leanprover/lean4:stable

  3.  Verify:
        lean --version
        lake --version

  4.  Create (or reuse) a Lean project directory that imports Mathlib:
        mkdir lean && cd lean
        lake init LeanProject math
        lake build

  The project expects a 'lean/' directory with a working lakefile
  so that `lake env lean <file>` can compile individual files.
"""


def _which(name: str) -> str | None:
    return shutil.which(name)


def check_lean_env(toolchain_dir: str = "lean") -> None:
    """Raise RuntimeError with setup instructions if lean/lake are missing."""
    lean_path = _which("lean")
    lake_path = _which("lake")
    missing: list[str] = []
    if not lean_path:
        missing.append("lean")
    if not lake_path:
        missing.append("lake")

    if missing:
        raise RuntimeError(
            f"Missing executables: {', '.join(missing)}.\n\n"
            f"{LEAN_SETUP_INSTRUCTIONS}"
        )

    toolchain = Path(toolchain_dir).resolve()
    if not toolchain.is_dir():
        raise RuntimeError(
            f"Lean toolchain directory not found: {toolchain}\n"
            "Create it with:  mkdir lean && cd lean && lake init LeanProject math && lake build"
        )

    lakefile = toolchain / "lakefile.lean"
    lakefile_toml = toolchain / "lakefile.toml"
    if not lakefile.exists() and not lakefile_toml.exists():
        raise RuntimeError(
            f"No lakefile found in {toolchain}.\n"
            "Run:  cd lean && lake init LeanProject math"
        )

    print(f"[lean_env] lean  = {lean_path}", file=sys.stderr)
    print(f"[lean_env] lake  = {lake_path}", file=sys.stderr)
    print(f"[lean_env] toolchain dir = {toolchain}", file=sys.stderr)


def lean_version() -> str:
    """Return the output of ``lean --version``, or an error string."""
    try:
        r = subprocess.run(
            ["lean", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.stdout.strip() or r.stderr.strip()
    except Exception as exc:
        return f"(could not determine lean version: {exc})"
