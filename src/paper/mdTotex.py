#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-oriented Markdown to LaTeX conversion (optimization research papers)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

STREAM_REQUIRED: Optional[bool] = None


# =========================
# LLM Prompts
# =========================

TAG_RECOVERY_PROMPT = (
    "Align equation numbers for ONE page of an optimization research paper.\n"
    "Inputs are for the same page:\n"
    "A) OCR Markdown\n"
    "B) PDF plain text\n"
    "\n"
    "Task: ONLY insert missing LaTeX \\\\tag{...} into matching DISPLAY equations.\n"
    "\n"
    "Hard rules:\n"
    "1) Keep text exactly as-is: no rewrite, no reorder, no paraphrase.\n"
    "2) Allowed edit is ONLY adding \\\\tag{...}.\n"
    "3) Never modify existing tags; never duplicate tags.\n"
    "4) If matching is uncertain, skip that equation.\n"
    "5) Output Markdown content only. No explanation, no fences.\n"
    "6) Never output any instruction text.\n"
    "\n"
    "<<<MD>>>\n"
)

LATEX_CONVERT_PROMPT = (
    "Convert OCR Markdown to LaTeX for an optimization research paper.\n"
    "Output ONLY LaTeX content. No commentary. No code fences.\n"
    "\n"
    "Placeholders:\n"
    "- Tokens like ZZZ_MATHBLOCK_0001_ZZZ are immutable placeholders.\n"
    "- Keep each token EXACTLY unchanged, on its own line, same order.\n"
    "- Do not move, wrap, or alter placeholders.\n"
    "\n"
    "Content rules:\n"
    "- Preserve wording, order, and paragraph boundaries.\n"
    "- Keep theorem/lemma/definition/proposition/corollary/remark numbering exactly.\n"
    "- Keep algorithm pseudocode, figure/table captions, and citation markers.\n"
    "- Keep section/subsection headings, authors, affiliation, abstract, and keywords.\n"
    "- Do not summarize, expand, or duplicate lines.\n"
    "- Do not output instruction/prompt text.\n"
    "\n"
    "Environment rules:\n"
    "- Only create theorem-like environments when the input explicitly has numbered markers\n"
    "  (Theorem/Lemma/Proposition/Corollary/Definition/Remark + number).\n"
    "- Only create \\\\begin{proof}...\\\\end{proof} when explicit 'Proof.' marker exists.\n"
    "- Use \\\\begin{algorithm}...\\\\end{algorithm} for pseudocode blocks when present.\n"
    "- Do not invent definitions, proofs, sections, or labels.\n"
    "\n"
    "Math rules:\n"
    "- Inline math uses $...$.\n"
    "- Display math uses \\\\[ ... \\\\] unless an existing \\\\tag requires equation/align-like env.\n"
    "- Keep existing \\\\tag only; never invent new \\\\tag.\n"
    "- Keep formulas faithful; do not alter meaning.\n"
    "\n"
    "Markdown:\n"
)

PROOF_SPLIT_PROMPT = (
    "You are given OCR Markdown from an optimization paper that contains an explicit 'Proof.' marker.\n"
    "The chunk may include non-proof text after the proof ends.\n"
    "\n"
    "Placeholders like ZZZ_MATHBLOCK_0001_ZZZ are immutable:\n"
    "- Keep unchanged, same order, on their own lines.\n"
    "- Do not move or wrap placeholders.\n"
    "\n"
    "Task:\n"
    "1) Convert to LaTeX.\n"
    "2) Split into (proof part) and (trailing non-proof part).\n"
    "3) Output in EXACT format:\n"
    "<<<PROOF>>>\n"
    "<LaTeX proof env only: must contain \\\\begin{proof} ... \\\\end{proof}>\n"
    "<<<REST>>>\n"
    "<Remaining LaTeX after proof ends; empty if none>\n"
    "\n"
    "Rules:\n"
    "- Output only the required two blocks; no commentary.\n"
    "- Keep wording and order; no rewrite.\n"
    "- Do not invent theorem/definition/section envs.\n"
    "- Keep existing \\\\tag only; never invent new \\\\tag.\n"
    "- If proof end is clear, move following text into <<<REST>>>.\n"
    "- Never output instruction text.\n"
    "\n"
    "Markdown:\n"
)


# =========================
# Config helpers
# =========================

def find_config_json() -> Path:
    p = Path.cwd() / "config.json"
    if p.exists():
        return p.resolve()
    here = Path(__file__).resolve().parent
    for d in [here] + list(here.parents):
        q = d / "config.json"
        if q.exists():
            return q.resolve()
    raise FileNotFoundError("config.json not found (checked CWD and script parents).")


def load_config() -> Dict[str, Any]:
    cfg_path = find_config_json()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a JSON object.")
    return data


def require_str(cfg: Dict[str, Any], key: str) -> str:
    v = cfg.get(key)
    if not isinstance(v, str) or not v.strip():
        raise KeyError(f"Missing/invalid '{key}' in config.json (must be non-empty string)")
    return v.strip()


def get_cfg(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def find_settings_json() -> Path:
    p = Path(__file__).resolve().with_name("settings.json")
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"settings.json not found next to script: {p}")


def load_settings() -> Dict[str, Any]:
    sp = find_settings_json()
    data = json.loads(sp.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{sp} must contain a JSON object.")
    return data


def get_setting(settings: Dict[str, Any], key: str, default: Any) -> Any:
    return settings.get(key, default)


def project_root_from_config() -> Path:
    return find_config_json().parent


# =========================
# Stream helpers
# =========================

def _coerce_stream_mode(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
        if s in {"auto", ""}:
            return None
    return None


def _requires_stream_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "stream must be set to true" in msg


def _collect_stream_text(stream) -> str:
    parts: List[str] = []
    for event in stream:
        try:
            choices = getattr(event, "choices", None)
            if not choices:
                continue
            choice0 = choices[0]
            delta = getattr(choice0, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if content:
                    parts.append(content)
                continue
            msg = getattr(choice0, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if content:
                    parts.append(content)
        except Exception:
            continue
    return "".join(parts)


def _chat_complete_text(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stream_mode: Optional[bool] = None,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    def _call_non_stream() -> str:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _call_stream() -> str:
        resp = client.chat.completions.create(**kwargs, stream=True)
        return _collect_stream_text(resp)

    mode = STREAM_REQUIRED if stream_mode is None else stream_mode
    if mode is True:
        return _call_stream()
    if mode is False:
        return _call_non_stream()

    try:
        return _call_non_stream()
    except Exception as e:
        if _requires_stream_error(e):
            return _call_stream()
        raise


# =========================
# OCR Markdown sanitization
# =========================

_MD_LEAK_LINE_PATTERNS = [
    r"(?i)the output must be latex",
    r"(?i)output only latex",
    r"(?i)strict block rules",
    r"(?i)the input may contain placeholder tokens",
    r"(?i)you must keep every placeholder token exactly unchanged",
    r"(?i)do not move placeholders",
    r"(?i)critical:\s*do not invent any new \\tag",
    r"(?i)^markdown:\s*$",
    r"(?i)^math rules:\s*$",
    r"(?i)^task:\s*$",
    r"(?i)^hard rules",
    r"(?i)^<<<proof>>>$",
    r"(?i)^<<<rest>>>$",
]


def _strip_md_instruction_leakage(md: str) -> str:
    lines = (md or "").splitlines()
    out: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if any(re.search(p, s) is not None for p in _MD_LEAK_LINE_PATTERNS):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _drop_runaway_number_lines(md: str) -> str:
    out: List[str] = []
    for ln in (md or "").splitlines():
        if re.search(r"(?:\b\d+\.\s*){80,}", ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _squash_repeated_lines(md: str, *, max_run: int = 2, min_len: int = 24) -> str:
    lines = (md or "").splitlines()
    out: List[str] = []
    prev_key = None
    run = 0
    for ln in lines:
        key = re.sub(r"\s+", " ", (ln or "").strip()).lower()
        comparable = len(key) >= min_len
        if comparable and key == prev_key:
            run += 1
        else:
            prev_key = key if comparable else None
            run = 1
        if comparable and run > max_run:
            continue
        out.append(ln)
    return "\n".join(out).strip()


def sanitize_ocr_markdown(md: str) -> str:
    """
    Defensive pre-clean for OCR markdown:
    - strip leaked instruction lines from prompts
    - drop obvious runaway number-loop lines
    - squash long repeated line runs
    """
    s = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _strip_md_instruction_leakage(s)
    s = _drop_runaway_number_lines(s)
    s = _squash_repeated_lines(s, max_run=2, min_len=24)
    return s.strip()


# =========================
# Markdown: page split
# =========================

PAGE_SPLIT_RE = re.compile(r"(?m)^\s*<!--\s*PAGE\s+(\d+)\s*-->\s*$")


def split_markdown_pages(md_text: str) -> List[Tuple[int, str]]:
    """
    Split a merged multi-page Markdown (produced by pdfTomd.py) into
    a list of (page_number, page_content) pairs.

    pdfTomd.py inserts <!-- PAGE N --> markers between pages.
    If no such markers are found, treat the whole document as page 1.
    """
    parts = PAGE_SPLIT_RE.split(md_text)
    out: List[Tuple[int, str]] = []
    i = 1
    while i + 1 < len(parts):
        page_num = int(parts[i])
        content = (parts[i + 1] or "").strip()
        out.append((page_num, content))
        i += 2
    if not out:
        out = [(1, (md_text or "").strip())]
    return out


# =========================
# PDF page text extraction (for tag recovery)
# =========================

def extract_pdf_page_text(pdf_path: Path, page_1based: int) -> str:
    """Extract plain text from a single PDF page via PyMuPDF."""
    if fitz is None:
        return ""
    try:
        doc = fitz.open(pdf_path)
        idx = page_1based - 1
        if idx < 0 or idx >= doc.page_count:
            return ""
        page = doc.load_page(idx)
        txt = page.get_text("text") or ""
        doc.close()
        return txt.strip()
    except Exception:
        return ""


def infer_pdf_path(in_md: Path, pdf_arg: Optional[str]) -> Optional[Path]:
    """
    Try to find the corresponding PDF for tag recovery.

    Priority:
    1) --pdf argument (explicit path)
    2) Infer from project structure:
         <root>/work/paper/<stem>/<stem>.md  ->  <root>/input_pdfs/paper/<stem>.pdf
    3) Fallback: look for <stem>.pdf next to the Markdown file
    """
    if pdf_arg:
        p = Path(pdf_arg).expanduser()
        if p.exists():
            return p.resolve()

    root = project_root_from_config()
    work_dir = root / "work" / "paper"
    input_dir = root / "input_pdfs" / "paper"

    try:
        rel = in_md.resolve().relative_to(work_dir.resolve())
        stem = in_md.stem
        # Expected layout: work/paper/<stem>/<stem>.md
        if len(rel.parts) >= 2 and rel.parts[-2] == stem and rel.parts[-1] == f"{stem}.md":
            cand = input_dir / f"{stem}.pdf"
            if cand.exists():
                return cand.resolve()
    except Exception:
        pass

    # Last resort: local search
    for cand in [in_md.with_suffix(".pdf"), in_md.parent / f"{in_md.stem}.pdf"]:
        if cand.exists():
            return cand.resolve()

    return None


# =========================
# Page-wise tag recovery
# =========================

def _normalize_for_compare(s: str) -> str:
    """Compare texts while ignoring whitespace noise and tag insertions."""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\\tag\{[^}]*\}", "", s)
    lines = []
    for ln in s.splitlines():
        ln2 = re.sub(r"\s+", " ", ln.strip())
        lines.append(ln2)
    return "\n".join(lines).strip()


def _similar_enough(a: str, b: str, *, threshold: float = 0.995) -> bool:
    a2 = _normalize_for_compare(a)
    b2 = _normalize_for_compare(b)
    if a2 == b2:
        return True
    ratio = SequenceMatcher(None, a2, b2).ratio()
    return ratio >= threshold


def _truncate_pdf_text(s: str, max_chars: int = 12000) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    head = s[: max_chars // 2]
    tail = s[-max_chars // 2 :]
    return head + "\n...\n" + tail


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=12),
    retry=retry_if_exception_type(Exception),
)
def llm_recover_page_tags(
    client: OpenAI,
    model: str,
    md_page: str,
    pdf_text: str,
    max_tokens: int,
) -> str:
    prompt = (
        TAG_RECOVERY_PROMPT
        + (md_page or "").strip()
        + "\n<<<PDF_TEXT>>>\n"
        + _truncate_pdf_text(pdf_text)
        + "\n"
    )
    text = _chat_complete_text(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return (text or "").strip()


def pagewise_tag_recovery(
    pages: List[Tuple[int, str]],
    pdf_path: Optional[Path],
    client: OpenAI,
    tag_model: str,
    max_tokens_tag: int,
    workers: int,
) -> List[Tuple[int, str]]:
    """
    For each page, recover missing equation numbers by inserting \\tag{...}.
    If pdf_path is None or extraction fails, returns original pages unchanged.

    Safety gate: only accept LLM output that is "mostly identical" to the
    original (ignoring added tags), using a high similarity threshold.
    """
    if pdf_path is None or not pdf_path.exists():
        return pages

    # Extract all PDF page text first (IO-bound)
    pdf_text_by_page: Dict[int, str] = {}
    for page_num, _md in pages:
        pdf_text_by_page[page_num] = extract_pdf_page_text(pdf_path, page_num)

    out: Dict[int, str] = {}

    def _process_one(page_num: int, md_page: str) -> Tuple[int, str]:
        pdf_text = pdf_text_by_page.get(page_num, "")
        if not md_page.strip() or not pdf_text.strip():
            return page_num, md_page

        fixed = llm_recover_page_tags(
            client=client,
            model=tag_model,
            md_page=md_page,
            pdf_text=pdf_text,
            max_tokens=max_tokens_tag,
        )

        if fixed and _similar_enough(md_page, fixed):
            return page_num, fixed.strip()

        return page_num, md_page

    workers2 = max(1, int(workers or 1))
    with ThreadPoolExecutor(max_workers=workers2) as ex:
        futs = {ex.submit(_process_one, p, md): p for p, md in pages}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Tag recovery (page-wise)"):
            p, fixed = fut.result()
            out[p] = fixed

    return [(p, out.get(p, md)) for p, md in pages]


# =========================
# Main (entry layer)
# =========================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert OCR Markdown of an optimization paper to LaTeX."
    )
    ap.add_argument("in_md", type=str, help="Input Markdown file (produced by pdfTomd.py)")
    ap.add_argument("out_tex", type=str, help="Output LaTeX .tex file path")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override LLM worker concurrency (default: read from settings.json)",
    )
    ap.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Optional explicit path to the source PDF (used for page-wise tag recovery)",
    )
    args = ap.parse_args()

    # ---- Load config & settings ----
    cfg = load_config()
    settings = load_settings()

    api_key = require_str(cfg, "api_key")
    base_url = get_cfg(cfg, "base_url", "https://aihubmix.com/v1")
    model = require_str(cfg, "model")

    global STREAM_REQUIRED
    stream_mode = _coerce_stream_mode(get_cfg(cfg, "stream", None))
    if stream_mode is None:
        stream_mode = _coerce_stream_mode(get_cfg(cfg, "force_stream", None))
    STREAM_REQUIRED = stream_mode

    max_tokens_think = int(get_setting(settings, "MDTOTEX_MAX_TOKENS", 2048))
    max_tokens_tag   = int(get_setting(settings, "MDTOTEX_MAX_TOKENS_TAG", 1024))
    timeout          = int(get_setting(settings, "MDTOTEX_TIMEOUT", 120))

    workers: Optional[int] = args.workers
    if workers is None or workers <= 0:
        workers = int(get_setting(settings, "MDTOTEX_WORKERS", 4))

    # ---- Resolve paths ----
    in_md = Path(args.in_md).expanduser().resolve()
    out_tex = Path(args.out_tex).expanduser().resolve()

    if not in_md.exists():
        print(f"ERROR: Markdown not found: {in_md}", file=sys.stderr)
        sys.exit(2)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # ---- OpenAI client ----
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    # ---- Preprocessing layer ----

    # (1) Read & sanitize raw OCR Markdown
    md_text = in_md.read_text(encoding="utf-8")
    md_text = sanitize_ocr_markdown(md_text)

    # (2) Split into (page_num, page_content) pairs
    pages = split_markdown_pages(md_text)
    print(f"[preprocess] pages detected: {len(pages)}")

    # (3) Locate corresponding PDF for tag recovery
    pdf_path = infer_pdf_path(in_md, args.pdf)
    if pdf_path is None:
        print("[tag] PDF not found -> skip page-wise tag recovery")
    else:
        print(f"[tag] PDF -> {pdf_path}")

    # (4) Page-wise tag recovery (inserts missing \\tag{...} in display equations)
    pages_fixed = pagewise_tag_recovery(
        pages=pages,
        pdf_path=pdf_path,
        client=client,
        tag_model=model,
        max_tokens_tag=max_tokens_tag,
        workers=workers,
    )

    # (5) Merge pages back into a single cleaned Markdown string
    full_md = "\n\n".join(
        md for _p, md in sorted(pages_fixed, key=lambda x: x[0])
    ).strip()
    full_md = sanitize_ocr_markdown(full_md)

    print(
        f"[preprocess] done — {len(pages_fixed)} pages, "
        f"{len(full_md)} chars, workers={workers}"
    )

    # TODO: structural layer (heading injection, greedy chunking) — next step
    # TODO: conversion layer (LLM block conversion)               — next step
    # TODO: post-fix / assembly layer                              — next step

    print(f"[stub] preprocessing complete. full_md length = {len(full_md)}")


if __name__ == "__main__":
    main()
