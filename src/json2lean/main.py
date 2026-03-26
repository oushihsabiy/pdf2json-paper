"""json2lean – convert JSON optimization exercises into Lean 4 files.

CLI entry-point and pipeline orchestrator.

Usage:
    python -m json2lean input.json [--output-dir outputs] [--no-preprocess]
                                    [--no-validate] [--no-recover]
                                    [--config config.json] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .api_client import APIClient
from .lean_env import check_lean_env, lean_version
from .loader import load_config, load_json, write_json
from .models import Exercise, ExerciseStatus, PipelineConfig
from .parser import parse_exercises
from .preprocessor import preprocess_all
from .translater import translate_all
from .validator import validate_all
from .writer import write_all
from .recover import recover_all


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="json2lean",
        description="Convert JSON optimization exercises into Lean 4 files.",
    )
    p.add_argument("input_json", help="Path to the input JSON file.")
    p.add_argument(
        "--output-dir", "-o", default="outputs",
        help="Directory for generated Lean files (default: outputs/).",
    )
    p.add_argument(
        "--log-dir", default="logs",
        help="Directory for logs and token-usage records (default: logs/).",
    )
    p.add_argument("--config", default=None, help="Path to config.json.")
    p.add_argument("--model", default=None, help="Override model from config.")
    p.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip the preprocessing (concise_to_lean) step.",
    )
    p.add_argument(
        "--no-validate", action="store_true",
        help="Skip Lean compilation / validation.",
    )
    p.add_argument(
        "--no-recover", action="store_true",
        help="Skip automatic recovery on validation failure.",
    )
    p.add_argument(
        "--max-recovery-retries", type=int, default=None,
        help="Override recovery retry limit from config.",
    )
    return p


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    # ---- Config ---------------------------------------------------
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)
    if args.model:
        cfg.model = args.model
    if args.max_recovery_retries is not None:
        cfg.recovery_max_retries = args.max_recovery_retries

    do_preprocess = cfg.preprocessing_enabled and not args.no_preprocess
    do_validate = not args.no_validate
    do_recover = not args.no_recover

    output_dir = Path(args.output_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- Lean environment check -----------------------------------
    if do_validate:
        check_lean_env(cfg.lean_toolchain_dir)
        print(f"[pipeline] Lean version: {lean_version()}", file=sys.stderr)

    # ---- Load input -----------------------------------------------
    input_path = Path(args.input_json).expanduser().resolve()
    data = load_json(input_path)
    exercises = parse_exercises(data)
    if not exercises:
        print("[pipeline] No exercises found in input file.", file=sys.stderr)
        sys.exit(1)
    print(f"[pipeline] Loaded {len(exercises)} exercise(s) from {input_path}",
          file=sys.stderr)

    # ---- API client -----------------------------------------------
    client = APIClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        timeout=cfg.timeout_seconds,
    )

    # ---- Step 1: Preprocess ---------------------------------------
    if do_preprocess:
        print("[pipeline] === Preprocessing ===", file=sys.stderr)
        preprocess_failed = preprocess_all(
            client, exercises,
            max_tokens=cfg.preprocessing_max_tokens,
            max_attempts=cfg.preprocessing_max_attempts,
        )
        if preprocess_failed:
            print(
                f"[pipeline] {len(preprocess_failed)} exercise(s) kept original problem.",
                file=sys.stderr,
            )
    else:
        print("[pipeline] Preprocessing skipped.", file=sys.stderr)

    # ---- Step 2: Translate ----------------------------------------
    print("[pipeline] === Translation ===", file=sys.stderr)
    translate_failed = translate_all(
        client, exercises,
        max_tokens=cfg.translation_max_tokens,
        max_attempts=cfg.translation_max_attempts,
    )

    # ---- Step 3: Write Lean files ---------------------------------
    print("[pipeline] === Writing Lean files ===", file=sys.stderr)
    written = write_all(exercises, output_dir)
    print(f"[pipeline] Wrote {len(written)} file(s) to {output_dir}", file=sys.stderr)

    # ---- Step 4: Validate -----------------------------------------
    if do_validate:
        print("[pipeline] === Validation ===", file=sys.stderr)
        validate_all(
            exercises, output_dir,
            toolchain_dir=cfg.lean_toolchain_dir,
            timeout=cfg.lean_timeout_seconds,
        )

        # ---- Step 5: Recover --------------------------------------
        broken = [e for e in exercises if not e.is_valid and e.status != ExerciseStatus.ERROR]
        if broken and do_recover:
            print("[pipeline] === Recovery ===", file=sys.stderr)
            still_broken = recover_all(
                client, exercises, output_dir,
                toolchain_dir=cfg.lean_toolchain_dir,
                lean_timeout=cfg.lean_timeout_seconds,
                max_tokens=cfg.recovery_max_tokens,
                max_retries=cfg.recovery_max_retries,
            )
            if still_broken:
                print(
                    f"[pipeline] {len(still_broken)} exercise(s) could not be repaired.",
                    file=sys.stderr,
                )
        elif broken:
            print(
                f"[pipeline] {len(broken)} exercise(s) failed validation (recovery skipped).",
                file=sys.stderr,
            )
    else:
        print("[pipeline] Validation skipped.", file=sys.stderr)

    # ---- Summary --------------------------------------------------
    _print_summary(exercises)

    # ---- Token usage log ------------------------------------------
    _save_token_log(client, exercises, log_dir)

    print("[pipeline] Done.", file=sys.stderr)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _print_summary(exercises: List[Exercise]) -> None:
    valid = sum(1 for e in exercises if e.status == ExerciseStatus.VALID)
    failed = sum(1 for e in exercises if e.status == ExerciseStatus.REPAIR_FAILED)
    errors = sum(1 for e in exercises if e.status == ExerciseStatus.ERROR)
    other = len(exercises) - valid - failed - errors
    print(
        f"\n[summary] total={len(exercises)}  valid={valid}  "
        f"repair_failed={failed}  error={errors}  other={other}",
        file=sys.stderr,
    )


def _save_token_log(client: APIClient, exercises: List[Exercise], log_dir: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"token_usage_{timestamp}.json"
    report = {
        "timestamp": timestamp,
        "total": client.total_usage(),
        "calls": client.dump_usage(),
        "exercises": [
            {
                "label": e.label,
                "status": e.status.value,
                "repair_attempts": e.repair_attempts,
                "num_errors": len(e.errors),
                "num_warnings": len(e.warnings),
            }
            for e in exercises
        ],
    }
    write_json(log_path, report)
    print(f"[pipeline] Token usage saved to {log_path}", file=sys.stderr)


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
