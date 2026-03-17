#!/usr/bin/env python3
"""Debug why a proof block ends early in mdTotex output.

This script traces internal LLM calls inside src.paper.mdTotex.markdown_to_latex
for one target proof block and reports:
- extracted line range and sizes
- placeholder sequence integrity
- whether fallback recursion happened (multiple LLM calls)
- where \\end{proof} first appears in generated LaTeX

Usage examples:
  python work/debug/test_proof_early_end.py --dry-run
  python work/debug/test_proof_early_end.py --max-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI  # type: ignore
from src.paper import mdTotex as mdt  # type: ignore


@dataclass
class CallTrace:
    idx: int
    max_tokens: int
    prompt_chars: int
    prompt_placeholders: List[str]
    response_chars: int
    response_placeholders: List[str]
    response_begin_proof: int
    response_end_proof: int


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _extract_block_by_markers(text: str, start_marker: str, end_marker: str) -> Tuple[str, int, int]:
    lines = text.splitlines()
    start_idx = -1
    end_idx = -1

    for i, ln in enumerate(lines):
        if start_marker in ln:
            start_idx = i
            break

    if start_idx < 0:
        raise ValueError(f"start marker not found: {start_marker!r}")

    for i in range(start_idx + 1, len(lines)):
        if end_marker in lines[i]:
            end_idx = i
            break

    if end_idx < 0:
        raise ValueError(f"end marker not found: {end_marker!r}")

    block = "\n".join(lines[start_idx:end_idx])
    return block, start_idx + 1, end_idx


def _find_placeholders(s: str) -> List[str]:
    return [m.group(0) for m in re.finditer(r"ZZZ_MATHBLOCK_\d+_ZZZ", s or "")]


def _count(pattern: str, s: str) -> int:
    return len(re.findall(pattern, s or ""))


def _load_runtime_config(cfg_path: Path, settings_path: Path) -> Tuple[str, str, str, int]:
    cfg = json.loads(_read_text(cfg_path))
    settings = json.loads(_read_text(settings_path))

    api_key = cfg.get("api_key", "").strip()
    base_url = cfg.get("base_url", "https://api.openai.com/v1").strip()
    model = cfg.get("model", "gpt-4o").strip()
    max_tokens = int(settings.get("MDTOTEX_MAX_TOKENS", 2048))

    if not api_key:
        raise ValueError("config.json missing non-empty 'api_key'")

    return api_key, base_url, model, max_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="Trace proof conversion and diagnose early \\end{proof}.")
    ap.add_argument(
        "--md",
        default=str(ROOT / "work/paper/Crouzeix-conjecture/Crouzeix-conjecture.md"),
        help="Input markdown path",
    )
    ap.add_argument("--start-marker", default="*Proof of Theorem 4*", help="Block start marker")
    ap.add_argument("--end-marker", default="## 6 Breakdown of regularity", help="Block end marker")
    ap.add_argument("--config", default=str(ROOT / "config.json"), help="config.json path")
    ap.add_argument(
        "--settings",
        default=str(ROOT / "src/paper/settings.json"),
        help="src/paper/settings.json path",
    )
    ap.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens for this test")
    ap.add_argument("--save-dir", default=str(ROOT / "output_json/proof_debug"), help="Debug output directory")
    ap.add_argument("--dry-run", action="store_true", help="Only run extraction + placeholder analysis")
    args = ap.parse_args()

    md_path = Path(args.md).resolve()
    cfg_path = Path(args.config).resolve()
    settings_path = Path(args.settings).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    md_text = _read_text(md_path)
    block, start_line, end_line = _extract_block_by_markers(md_text, args.start_marker, args.end_marker)

    md_in = mdt.sanitize_ocr_markdown(block)
    md_ph, mapping, seq = mdt.replace_display_math_with_placeholders(md_in)

    (save_dir / "proof_block_raw.md").write_text(block, encoding="utf-8")
    (save_dir / "proof_block_placeholder.md").write_text(md_ph, encoding="utf-8")

    print("=== Block Info ===")
    print(f"source: {md_path}")
    print(f"line range: {start_line}..{end_line - 1}")
    print(f"raw chars: {len(block)}")
    print(f"sanitized chars: {len(md_in)}")
    print(f"placeholder chars: {len(md_ph)}")
    print(f"placeholder count: {len(seq)}")
    print(f"raw '$$' count: {block.count('$$')}")
    print()

    if args.dry_run:
        print("Dry-run complete. Saved:")
        print(f"  - {save_dir / 'proof_block_raw.md'}")
        print(f"  - {save_dir / 'proof_block_placeholder.md'}")
        return 0

    api_key, base_url, model, max_tokens_setting = _load_runtime_config(cfg_path, settings_path)
    max_tokens = int(args.max_tokens) if args.max_tokens is not None else max_tokens_setting

    client = OpenAI(api_key=api_key, base_url=base_url)
    traces: List[CallTrace] = []

    orig_chat = mdt._chat_complete_text

    def traced_chat_complete_text(client: Any, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        idx = len(traces) + 1
        prompt = ""
        if messages:
            prompt = str(messages[0].get("content", ""))

        prompt_ph = _find_placeholders(prompt)
        max_tok = int(kwargs.get("max_tokens", 0) or 0)

        raw = orig_chat(client, model=model, messages=messages, **kwargs)
        raw_clean = mdt.strip_outer_document(mdt.strip_code_fences(raw))

        traces.append(
            CallTrace(
                idx=idx,
                max_tokens=max_tok,
                prompt_chars=len(prompt),
                prompt_placeholders=prompt_ph,
                response_chars=len(raw_clean),
                response_placeholders=_find_placeholders(raw_clean),
                response_begin_proof=_count(r"\\begin\{proof\}", raw_clean),
                response_end_proof=_count(r"\\end\{proof\}", raw_clean),
            )
        )
        return raw

    mdt._chat_complete_text = traced_chat_complete_text
    try:
        final_latex = mdt.markdown_to_latex(client, model, block, max_tokens=max_tokens)
    finally:
        mdt._chat_complete_text = orig_chat

    (save_dir / "proof_result_final.tex").write_text(final_latex, encoding="utf-8")

    if traces:
        report_lines: List[str] = []
        report_lines.append("=== LLM Call Trace ===")
        report_lines.append(f"model: {model}")
        report_lines.append(f"max_tokens(requested): {max_tokens}")
        report_lines.append(f"llm_call_count: {len(traces)}")
        report_lines.append("")

        for t in traces:
            mismatch = t.response_placeholders != t.prompt_placeholders
            report_lines.append(
                f"Call #{t.idx}: prompt_chars={t.prompt_chars}, response_chars={t.response_chars}, "
                f"max_tokens={t.max_tokens}"
            )
            report_lines.append(
                f"  placeholders: prompt={len(t.prompt_placeholders)}, "
                f"response={len(t.response_placeholders)}, mismatch={mismatch}"
            )
            report_lines.append(
                f"  proof tags in response: begin={t.response_begin_proof}, end={t.response_end_proof}"
            )

        if len(traces) > 1:
            report_lines.append("")
            report_lines.append("Inference: markdown_to_latex likely entered recursive fallback.")
            report_lines.append("Reason is typically placeholder mismatch in an earlier call.")

        first_end = re.search(r"\\end\{proof\}", final_latex)
        if first_end:
            prefix = final_latex[: first_end.start()]
            line_no = prefix.count("\n") + 1
            report_lines.append("")
            report_lines.append(f"first \\end{{proof}} in final_latex at line: {line_no}")

        report = "\n".join(report_lines) + "\n"
        print(report)
        (save_dir / "trace_report.txt").write_text(report, encoding="utf-8")

    print("Saved:")
    print(f"  - {save_dir / 'proof_block_raw.md'}")
    print(f"  - {save_dir / 'proof_block_placeholder.md'}")
    print(f"  - {save_dir / 'proof_result_final.tex'}")
    print(f"  - {save_dir / 'trace_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
