#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-oriented Markdown to LaTeX conversion (optimization research papers)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
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
# Display-math placeholders
# (needed by attach_standalone_equation_numbers)
# =========================

PLACEHOLDER_PREFIX = "ZZZ_MATHBLOCK_"
PLACEHOLDER_SUFFIX = "_ZZZ"

# Matches $$ ... $$, \[ ... \], and equation/align/gather/multline/flalign[*]
_DISPLAY_MATH_BLOCK_RE = re.compile(
    r"(?s)"
    r"(?<!\\)\$\$.*?\$\$"
    r"|(?<!\\)\\\[.*?\\\]"
    r"|\\begin\{(?P<env>(?:equation|align|gather|multline|flalign)\*?)\}.*?\\end\{(?P=env)\}"
)

_PLACEHOLDER_RE = re.compile(r"\bZZZ_MATHBLOCK_\d{4}_ZZZ\b")
_TAG_TOKEN_RE = re.compile(r"\\tag\*?\{[^}]*\}")


def _needs_aligned_wrapper(body: str) -> bool:
    s = body or ""
    if re.search(
        r"\\begin\{(?:aligned|alignedat|split|array|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\}",
        s,
    ):
        return False
    return "\\\\" in s or "&" in s


def _build_bracket_display_math(inner: str) -> str:
    body = (inner or "").strip()
    tags = [t.strip() for t in _TAG_TOKEN_RE.findall(body) if t.strip()]
    body_wo_tags = _TAG_TOKEN_RE.sub("", body).strip()
    body_wrapped = (
        "\\begin{aligned}\n" + body_wo_tags.strip() + "\n\\end{aligned}"
        if _needs_aligned_wrapper(body_wo_tags)
        else body_wo_tags
    )
    tag_text = "\n" + "\n".join(tags) if tags else ""
    return "\\[\n" + body_wrapped.strip("\n") + tag_text + "\n\\]"


def sanitize_display_math_block(block: str) -> str:
    """Normalise a display-math block to \\[...\\] or align*."""
    b = (block or "").strip()
    m = re.match(r"(?s)^\$\$(.*?)\$\$$", b)
    if m:
        return _build_bracket_display_math(m.group(1))
    m = re.match(r"(?s)^\\\[(.*?)\\\]$", b)
    if m:
        return _build_bracket_display_math(m.group(1))
    m = re.match(
        r"(?s)^\\begin\{(?P<env>(?:equation|gather|multline|flalign|align)\*?)\}(?P<body>.*)\\end\{(?P=env)\}$",
        b,
    )
    if m:
        env = m.group("env")
        body = (m.group("body") or "").strip("\n")
        base = env[:-1] if env.endswith("*") else env
        if base == "align":
            return "\\begin{align*}\n" + body + "\n\\end{align*}"
        return _build_bracket_display_math(body)
    return _build_bracket_display_math(b)


def replace_display_math_with_placeholders(
    markdown: str,
) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Replace display-math blocks with immutable placeholder tokens.
    
    Strategy:
    - For $$...$$: pair odd-numbered $$ with next $$ (simple pairing rule)
    - For \\[...\\]: use regex
    - For \\begin{...}\\end{...}: use regex
    """
    mapping: Dict[str, str] = {}
    seq: List[str] = []
    idx = 0
    
    md = markdown or ""
    result: List[str] = []
    pos = 0
    
    # Step 1: Find all unescaped $$ positions
    dollar_positions: List[int] = []
    for m in re.finditer(r"(?<!\\)\$\$", md):
        dollar_positions.append(m.start())
    
    # Step 2: Pair them: 0 with 1, 2 with 3, etc.
    dollar_pairs = []
    processed_ranges = set()  # ranges that have been paired
    
    for i in range(0, len(dollar_positions) - 1, 2):
        start_pos = dollar_positions[i]
        end_pos = dollar_positions[i + 1] + 2  # +2 to include closing $$
        dollar_pairs.append((start_pos, end_pos))
        processed_ranges.add((start_pos, end_pos))
    
    # Step 3: Find \[...\] and \begin{...}\end{...} blocks
    other_blocks = []
    for m in re.finditer(r"(?<!\\)\\\[.*?\\\]", md, re.DOTALL):
        other_blocks.append((m.start(), m.end(), "bracket", m.group(0)))
    for m in re.finditer(
        r"\\begin\{(?P<env>(?:equation|align|gather|multline|flalign)\*?)\}.*?\\end\{(?P=env)\}",
        md, re.DOTALL
    ):
        other_blocks.append((m.start(), m.end(), "begin", m.group(0)))
    
    # Step 4: Combine and sort all blocks by position
    all_blocks = []
    for start, end in dollar_pairs:
        all_blocks.append((start, end, "dollar", md[start:end]))
    for start, end, btype, raw in other_blocks:
        all_blocks.append((start, end, btype, raw))
    
    all_blocks.sort(key=lambda x: x[0])
    
    # Step 5: De-duplicate overlapping blocks, keeping the first one
    filtered_blocks = []
    for start, end, btype, raw in all_blocks:
        overlaps = False
        for fstart, fend, _, _ in filtered_blocks:
            if (start < fend and end > fstart):
                overlaps = True
                break
        if not overlaps:
            filtered_blocks.append((start, end, btype, raw))
    
    # Step 6: Process text and blocks in order
    pos = 0
    for start, end, btype, raw in filtered_blocks:
        # Add text before this block
        if start > pos:
            result.append(md[pos:start])
        
        # Add placeholder for this block
        idx += 1
        ph = f"{PLACEHOLDER_PREFIX}{idx:04d}{PLACEHOLDER_SUFFIX}"
        seq.append(ph)
        mapping[ph] = sanitize_display_math_block(raw)
        result.append("\n" + ph + "\n")
        pos = end
    
    # Add remaining text
    if pos < len(md):
        result.append(md[pos:])
    
    md2 = "".join(result)
    return md2, mapping, seq


def restore_display_math_placeholders(latex: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return latex or ""

    def _repl(m: re.Match) -> str:
        return mapping.get(m.group(0), m.group(0))

    return _PLACEHOLDER_RE.sub(_repl, latex or "")


def split_markdown_by_display_math(markdown: str) -> List[Tuple[str, str]]:
    """Split markdown into [(kind, segment)] where kind is 'text' or 'math'."""
    s = markdown or ""
    out: List[Tuple[str, str]] = []
    pos = 0
    for m in _DISPLAY_MATH_BLOCK_RE.finditer(s):
        if m.start() > pos:
            out.append(("text", s[pos : m.start()]))
        out.append(("math", m.group(0)))
        pos = m.end()
    if pos < len(s):
        out.append(("text", s[pos:]))
    return out


# =========================
# Markdown post-fix: attach standalone equation numbers like "(1.4)"
# =========================

_STANDALONE_EQNUM_LINE_RE = re.compile(r"^\s*\((\d+(?:\.\d+)*)\)\s*$")


def _extract_display_math_inner(block: str) -> str:
    b = (block or "").strip()
    m = re.match(r"(?s)^\$\$(.*?)\$\$$", b)
    if m:
        inner = m.group(1)
    else:
        m = re.match(r"(?s)^\\\[(.*?)\\\]$", b)
        if m:
            inner = m.group(1)
        else:
            m = re.match(
                r"(?s)^\\begin\{(?P<env>(?:equation|gather|multline|flalign|align)\*?)\}(?P<body>.*)\\end\{(?P=env)\}$",
                b,
            )
            inner = m.group("body") if m else b
    inner = _TAG_TOKEN_RE.sub("", inner or "").strip()
    m2 = re.match(r"(?s)^\\begin\{aligned\}(.*?)\\end\{aligned\}$", inner.strip())
    if m2:
        inner = m2.group(1).strip()
    return inner.strip()


def _merge_math_blocks_with_tag(math_blocks: List[str], tag_num: str) -> str:
    inners: List[str] = []
    for blk in math_blocks:
        inner = _extract_display_math_inner(blk)
        inner = re.sub(r"\\\\\s*$", "", inner).strip()
        if inner:
            inners.append(inner)
    if not inners:
        return sanitize_display_math_block(math_blocks[-1]) if math_blocks else ""
    if len(inners) == 1:
        body = inners[0]
        wrapped = (
            "\\begin{aligned}\n" + body + "\n\\end{aligned}"
            if _needs_aligned_wrapper(body)
            else body
        )
        return "\\[\n" + wrapped.strip("\n") + "\n" + rf"\tag{{{tag_num}}}" + "\n\\]"
    body = " \\\\\n".join(inners)
    return (
        "\\[\n\\begin{aligned}\n"
        + body
        + "\n\\end{aligned}\n"
        + rf"\tag{{{tag_num}}}"
        + "\n\\]"
    )


def attach_standalone_equation_numbers(markdown: str) -> str:
    """
    Convert standalone equation-number lines like '(1.4)' into \\tag{1.4} attached
    to the immediately preceding display math block(s).
    """
    segs = split_markdown_by_display_math(markdown)
    out: List[Tuple[str, str]] = []

    for kind, seg in segs:
        if kind != "text":
            out.append((kind, seg))
            continue

        for line in (seg or "").splitlines(True):
            m = _STANDALONE_EQNUM_LINE_RE.match(line.strip())
            if not m:
                out.append(("text", line))
                continue

            tag_num = m.group(1).strip()
            # Remove trailing blank texts
            while out and out[-1][0] == "text" and out[-1][1].strip() == "":
                out.pop()

            # Collect preceding consecutive math blocks
            collected: List[str] = []
            while out and out[-1][0] == "math":
                collected.insert(0, out.pop()[1])
                while out and out[-1][0] == "text" and out[-1][1].strip() == "":
                    out.pop()

            if not collected:
                out.append(("text", line))
                continue

            if any(
                rf"\tag{{{tag_num}}}" in blk or rf"\tag*{{{tag_num}}}" in blk
                for blk in collected
            ):
                for blk in collected:
                    out.append(("math", blk))
                continue

            merged = _merge_math_blocks_with_tag(collected, tag_num)
            if merged:
                out.append(("math", merged))

    return "".join(seg for _, seg in out)


# =========================
# Heading injection (structure anchoring)
# =========================

HEADING_START = "<!-- HEADING_START -->"
HEADING_END = "<!-- HEADING_END -->"

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Paper section numbers: "1", "1.1", "1.1.1" (no trailing dot) followed by a title.
# Unlike the book version we also match single-digit top-level: "1 Introduction".
NUMSEC_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")

# All-caps single-word headings common in papers: ABSTRACT, REFERENCES, ACKNOWLEDGEMENTS
_ALLCAPS_HEADING_RE = re.compile(
    r"^\s*(ABSTRACT|REFERENCES?|ACKNOWLEDGEMENTS?|APPENDIX|NOTATION|CONCLUSION|INTRODUCTION)\s*$",
    re.IGNORECASE,
)


def inject_heading_sentinels(full_md: str) -> str:
    """
    Scan merged Markdown and wrap headings with strong sentinels.

    Supported heading forms for optimization papers:
      - Markdown headings: #, ##, ...
      - Numeric section headings: "1", "1.1", "1.1.1" followed by a title on the same line
      - All-caps single-keyword headings: ABSTRACT, REFERENCES, etc.
    """
    lines = (full_md or "").splitlines()
    out: List[str] = []

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip("\n")
        stripped = ln.strip()

        # (A) Markdown headings: ## Introduction
        hm = MD_HEADING_RE.match(stripped)
        if hm:
            out.append(HEADING_START)
            out.append(stripped)
            out.append(HEADING_END)
            i += 1
            continue

        # (B) Numeric section lines: "1.1 Related Work" or "1 Introduction"
        nm = NUMSEC_LINE_RE.match(stripped)
        if nm:
            out.append(HEADING_START)
            out.append(stripped)
            out.append(HEADING_END)
            i += 1
            continue

        # (C) All-caps keyword headings: ABSTRACT, REFERENCES ...
        cm = _ALLCAPS_HEADING_RE.match(stripped)
        if cm:
            out.append(HEADING_START)
            out.append(stripped)
            out.append(HEADING_END)
            i += 1
            continue

        out.append(ln)
        i += 1

    return "\n".join(out).strip() + "\n"


# =========================
# Greedy chunking (logical chunking)
# =========================

# Theorem-like environments in optimization papers (includes Remark, Assumption, Conjecture)
_STMT_START_RE = re.compile(
    r"^\s*(?:[*_`> ]*)"
    r"(Theorem|Lemma|Proposition|Corollary|Definition|Remark|Assumption|Conjecture)s?\s+"
    r"([0-9]+(?:\.[0-9]+)*)"
    r"\s*\.?\s*(?:[*_` ]*)"
    r"(.*)$",
    re.IGNORECASE,
)

_PROOF_START_RE = re.compile(
    r"^\s*(?:[*_`> ]*)Proof\s*[:.]?\s*(?:[*_` ]*)"
    r"(.*)$",
    re.IGNORECASE,
)


def _normalize_stmt_line(
    line: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    raw = (line or "").strip()
    m_bold = re.match(r"^\*\*(.+?)\*\*$", raw)
    if not m_bold:
        return None, None, []

    # New rule: only bold-wrapped statement headers are treated as stmt starts.
    stmt_text = m_bold.group(1).strip()
    m = _STMT_START_RE.match(stmt_text)
    if not m:
        return None, None, []
    kind = m.group(1).title()
    num = m.group(2)
    tail = (m.group(3) or "").strip()

    env_map = {
        "Theorem": "thm",
        "Lemma": "lem",
        "Proposition": "prop",
        "Corollary": "cor",
        "Definition": "defn",
        "Remark": "rmk",
        "Assumption": "asm",
        "Conjecture": "conj",
    }
    env = env_map.get(kind, "thm")

    stmt_id = f"{kind} {num}"
    first = f"{stmt_id}."
    lines = [first]
    if tail:
        tail = tail.lstrip(":-\u2013\u2014. ").rstrip()
        if tail:
            lines.append(tail)
    return env, stmt_id, lines


def _normalize_proof_line(line: str) -> Tuple[bool, List[str]]:
    m = _PROOF_START_RE.match(line.strip())
    if not m:
        return False, []
    tail = (m.group(1) or "").strip()
    lines = ["Proof."]
    if tail:
        tail = tail.lstrip(":-\u2013\u2014. ").rstrip()
        if tail:
            lines.append(tail)
    return True, lines


@dataclass
class Block:
    kind: str           # "heading" | "stmt" | "proof" | "para"
    env: Optional[str]  # for stmt: thm/lem/prop/cor/defn/rmk/asm/conj; for proof: "proof"
    md: str             # markdown payload


def greedy_chunk_markdown(anchored_md: str) -> List[Block]:
    """
    Streaming greedy scan over sentinel-annotated Markdown.

    Priority:
      1) Heading sentinel blocks: forced break, emitted as heading Block.
      2) Theorem-like env starts (stmt): forced break.
      3) Proof starts: forced break; proof blocks are greedy (extend to next heading/env-start).
      4) Everything else accumulates into 'para' blocks.
    """
    lines = (anchored_md or "").splitlines()
    blocks: List[Block] = []

    cur_kind: Optional[str] = None
    cur_env: Optional[str] = None
    cur_stmt_id: Optional[str] = None
    cur_lines: List[str] = []

    def flush() -> None:
        nonlocal cur_kind, cur_env, cur_stmt_id, cur_lines
        if cur_kind is None:
            return
        text = "\n".join(cur_lines).strip()
        if text:
            blocks.append(Block(kind=cur_kind, env=cur_env, md=text))
        cur_kind = None
        cur_env = None
        cur_stmt_id = None
        cur_lines = []

    i = 0
    while i < len(lines):
        ln = lines[i]

        # (A) Heading block
        if ln.strip() == HEADING_START:
            flush()
            i += 1
            heading_lines: List[str] = []
            while i < len(lines) and lines[i].strip() != HEADING_END:
                heading_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == HEADING_END:
                i += 1
            heading_text = "\n".join(heading_lines).strip()
            if heading_text:
                blocks.append(Block(kind="heading", env=None, md=heading_text))
            continue

        # (B) Theorem-like stmt start
        env, stmt_id, norm_stmt_lines = _normalize_stmt_line(ln)
        if env is not None:
            if cur_kind == "stmt" and cur_stmt_id and stmt_id == cur_stmt_id:
                # Repeated title across pages -> treat as continuation
                if len(norm_stmt_lines) > 1:
                    cur_lines.extend(norm_stmt_lines[1:])
                i += 1
                continue
            flush()
            cur_kind = "stmt"
            cur_env = env
            cur_stmt_id = stmt_id
            cur_lines = list(norm_stmt_lines)
            i += 1
            continue

        # (C) Proof start
        is_proof, norm_proof_lines = _normalize_proof_line(ln)
        if is_proof:
            if cur_kind == "proof":
                # Repeated 'Proof.' -> continuation
                if len(norm_proof_lines) > 1:
                    cur_lines.extend(norm_proof_lines[1:])
                i += 1
                continue
            flush()
            cur_kind = "proof"
            cur_env = "proof"
            cur_stmt_id = None
            cur_lines = list(norm_proof_lines)
            i += 1
            continue

        # (D) Normal line
        if cur_kind is None:
            cur_kind = "para"
            cur_env = None
            cur_stmt_id = None
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
        i += 1

    flush()
    return blocks


# =========================
# Heading -> LaTeX (deterministic)
# =========================

def heading_block_to_latex(heading_md: str) -> str:
    """
    Convert a heading block (content between HEADING_START/END) to a LaTeX heading command.

    Supported forms:
      - Numeric: "1 Introduction"  -> \\section{...}
                 "1.1 Background"  -> \\subsection{...}
                 "1.1.1 Details"   -> \\subsubsection{...}
      - Markdown: # / ## / ### etc.
      - All-caps keywords: ABSTRACT -> \\section*{Abstract}, REFERENCES -> \\section*{References}, etc.
    """
    lines = [ln.strip() for ln in (heading_md or "").splitlines() if ln.strip()]
    if not lines:
        return ""

    first = lines[0]

    # All-caps keyword headings
    cm = _ALLCAPS_HEADING_RE.match(first)
    if cm:
        title = first.title()  # "ABSTRACT" -> "Abstract"
        return rf"\section*{{{title}}}"

    # Markdown heading (# / ## / ...)
    hm = MD_HEADING_RE.match(first)
    if hm:
        level = len(hm.group(1))
        text = hm.group(2).strip()
        cmds = ["section", "subsection", "subsubsection", "paragraph", "subparagraph", "subparagraph"]
        cmd = cmds[min(level - 1, len(cmds) - 1)]
        return rf"\{cmd}{{{text}}}"

    # Numeric section: "1 Introduction" or "1.1 Background"
    nm = NUMSEC_LINE_RE.match(first)
    if nm:
        num = nm.group(1)
        title = nm.group(2).strip()
        # Guard against false positives from matrix rows like "0 & 2\\".
        if not re.match(r"^[A-Za-z][A-Za-z0-9\-\s,:;()]*$", title):
            return first
        depth = num.count(".")
        if depth == 0:
            cmd = "section"
        elif depth == 1:
            cmd = "subsection"
        elif depth == 2:
            cmd = "subsubsection"
        else:
            cmd = "paragraph"
        return rf"\{cmd}{{{num} {title}}}"

    # Fallback: treat as \section
    return rf"\section{{{first}}}"


# =========================
# Block size limiter
# =========================

def split_large_para_blocks(blocks: List[Block], max_chars: int = 3000) -> List[Block]:
    """
    Prevent LLM truncation by splitting oversized 'para' blocks at blank-line boundaries.
    Non-para blocks (stmt/proof/heading) are never split.
    """
    max_chars = max(500, int(max_chars or 3000))
    out: List[Block] = []
    for blk in blocks:
        if blk.kind != "para" or len(blk.md) <= max_chars:
            out.append(blk)
            continue
        parts = re.split(r"\n\s*\n+", blk.md.strip())
        buf: List[str] = []
        cur = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            add_len = len(part) + (2 if buf else 0)
            if buf and (cur + add_len) > max_chars:
                out.append(Block(kind="para", env=None, md="\n\n".join(buf).strip() + "\n"))
                buf = [part]
                cur = len(part)
            else:
                cur += add_len
                buf.append(part)
        if buf:
            out.append(Block(kind="para", env=None, md="\n\n".join(buf).strip() + "\n"))
    return out


# =========================
# Conversion layer: LLM-based Markdown→LaTeX
# =========================

class LowQualityLLMOutput(Exception):
    """Raised when LLM output fails validation (leaked prompts, excessive repetition)."""
    pass


# ---- Utility functions for LaTeX processing ----

def strip_code_fences(s: str) -> str:
    """Remove markdown code fences from LLM output."""
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:latex)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def strip_outer_document(s: str) -> str:
    """Remove \\documentclass ... \\end{document} wrapper (LLM sometimes adds this)."""
    s = (s or "").strip()
    s = re.sub(r"(?s)\\documentclass.*?\\begin\{document\}", "", s)
    s = re.sub(r"(?s)\\end\{document\}\s*$", "", s)
    return s.strip()


def normalize_unicode_symbols(latex: str) -> str:
    """Normalize Unicode symbols that should be LaTeX."""
    latex = latex.replace("§", r"\S ")
    latex = latex.replace("\u00A0", " ")
    return latex


def normalize_display_math(latex: str) -> str:
    """Normalize display math delimiters: $$...$$ -> \\[...\\]."""
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        return "\\[\n" + inner + "\n\\]"
    return re.sub(r"(?s)\$\$(.*?)\$\$", repl, latex)


def normalize_double_backslash_begin_end(latex: str) -> str:
    """Fix LLM bug: \\begin{...} should be \\begin{...}."""
    s = latex or ""
    s = re.sub(r"(?m)^(\s*)\\\\(begin|end)\{", r"\1\\\2{", s)
    s = re.sub(r"(?<!\\)\\\\(begin|end)\{", r"\\\1{", s)
    return s


def _is_hard_boundary(line: str) -> bool:
    """Check if a line is a structural boundary (section, clearpage, etc)."""
    s = (line or "").lstrip()
    if not s:
        return False
    if s.startswith((r"\section", r"\subsection", r"\subsubsection", r"\chapter", r"\paragraph",
                      r"\clearpage", r"\newpage")):
        return True
    if s.startswith("% ===== Page"):
        return True
    return False


def _balance_math_env_pairs(latex: str) -> str:
    """
    Heal common LLM mistakes in equation environments:
    - duplicate \\begin{equation}
    - illegal nesting (equation inside aligned)
    - missing \\end{...}
    
    Conservative line-oriented approach.
    """
    lines = (latex or "").splitlines()
    out: List[str] = []
    stack: List[str] = []
    
    _MATH_ENV_RE = re.compile(r"^\s*\\(?P<kind>begin|end)\{(?P<env>(?:equation|align|gather|multline|flalign|align\*|equation\*)\*?)\}\s*$")
    
    def prev_nonempty() -> str:
        for k in range(len(out) - 1, -1, -1):
            t = out[k].strip()
            if t:
                return t
        return ""
    
    for line in lines:
        # If at hard boundary, close any open envs
        if stack and _is_hard_boundary(line):
            while stack:
                out.append(rf"\end{{{stack.pop()}}}")
        
        m = _MATH_ENV_RE.match(line)
        if not m:
            out.append(line)
            continue
        
        kind = m.group("kind")
        env = m.group("env")
        
        if kind == "begin":
            # Close any open env before starting new one (prevent nesting)
            if stack:
                open_env = stack.pop()
                out.append(rf"\end{{{open_env}}}")
            stack.append(env)
            out.append(line)
            continue
        
        # kind == "end"
        if stack:
            stack.pop()
            out.append(line)
        # else: drop unmatched end to keep doc compilable
    
    # Close any remaining open envs
    while stack:
        out.append(rf"\end{{{stack.pop()}}}")
    
    return "\n".join(out)


def _normalize_opt_operators(latex: str) -> str:
    """Normalize OCR artifacts like \\minimize -> \\min."""
    latex = re.sub(r"\\minimize\b", r"\\min", latex)
    latex = re.sub(r"\\maximize\b", r"\\max", latex)
    return latex


def sanitize_latex_math(latex: str) -> str:
    """
    Perform basic LaTeX sanitization for math content.
    Simpler than book version (papers usually have cleaner OCR).
    """
    s = latex or ""
    s = _normalize_opt_operators(s)
    s = normalize_display_math(s)
    s = normalize_double_backslash_begin_end(s)
    s = _balance_math_env_pairs(s)
    return s.strip()


def heal_latex_fragment(latex: str) -> str:
    """
    Core healing pipeline for LaTeX fragments.
    Minimal but essential fixes for LLM-generated LaTeX.
    """
    latex = strip_code_fences(latex)
    latex = strip_outer_document(latex)
    latex = normalize_unicode_symbols(latex)
    latex = sanitize_latex_math(latex)
    return latex.strip()


# ---- Validation functions ----

def _has_prompt_leak(s: str) -> bool:
    """Detect if LLM output contains leaked prompt text."""
    tell_tale = [
        "the output must be latex",
        "output only latex",
        "convert ocr markdown",
        "placeholders like zzz",
        "do not invent",
        "hard rules",
        "environment rules",
    ]
    s_lower = (s or "").lower()
    return any(tell in s_lower for tell in tell_tale)


def _has_pathological_repetition(s: str) -> bool:
    """Detect pathological repetition in LLM output (sign of failure)."""
    lines = (s or "").splitlines()
    if len(lines) < 5:
        return False
    
    # Check if >40% of lines are identical
    line_counts: Dict[str, int] = {}
    for ln in lines:
        normalized = re.sub(r"\s+", " ", ln.strip()).lower()
        if len(normalized) > 20:  # Skip short lines
            line_counts[normalized] = line_counts.get(normalized, 0) + 1
    
    if not line_counts:
        return False
    
    max_count = max(line_counts.values())
    if max_count > len(lines) * 0.4:
        return True
    
    return False


def _validate_llm_tex_output(raw: str) -> None:
    """Validate LLM output for common failure modes."""
    if _has_prompt_leak(raw):
        raise LowQualityLLMOutput("LLM output leaked prompt text.")
    if _has_pathological_repetition(raw):
        raise LowQualityLLMOutput("LLM output is pathologically repetitive.")


# ---- Core LLM conversion ----

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(Exception),
)
def markdown_to_latex(client: OpenAI, model: str, markdown: str, max_tokens: int) -> str:
    """
    Convert a Markdown block to LaTeX using LLM, preserving display-math placeholders.
    
    Strategy:
      1) Replace display-math blocks with stable placeholders
      2) Send to LLM for conversion, keeping placeholders unchanged
      3) Restore the math blocks
      4) Heal common LaTeX issues
    """
    md_in = sanitize_ocr_markdown(markdown or "")
    if not md_in.strip():
        return ""
    
    # Placeholder preservation
    md_ph, mapping, seq = replace_display_math_with_placeholders(md_in)
    prompt = LATEX_CONVERT_PROMPT + md_ph.strip()
    
    # LLM call
    raw = _chat_complete_text(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    
    # Clean and validate
    raw = strip_code_fences(raw)
    raw = strip_outer_document(raw)
    _validate_llm_tex_output(raw)
    
    # Restore math placeholders
    if mapping:
        found = [match.group(0) for match in re.finditer(r"ZZZ_MATHBLOCK_\d+_ZZZ", raw)]
        if found != seq:
            # Fallback: if placeholders were corrupted, try segment-by-segment conversion
            parts: List[str] = []
            for kind, seg in split_markdown_by_display_math(md_in):
                if kind == "math":
                    parts.append(sanitize_display_math_block(seg))
                else:
                    t = (seg or "").strip()
                    if t:
                        parts.append(markdown_to_latex(client, model, t, max_tokens))
            merged = "\n\n".join([p for p in parts if p and p.strip()])
            return heal_latex_fragment(merged)
        
        raw = restore_display_math_placeholders(raw, mapping)
    
    return heal_latex_fragment(raw)


def _parse_proof_split_response(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    t = strip_code_fences(t)
    t = strip_outer_document(t)
    m = re.search(r"(?s)<<<PROOF>>>\s*(.*?)\s*<<<REST>>>\s*(.*)\s*$", t)
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()


def _split_proof_output(text: str) -> Tuple[str, str]:
    return _parse_proof_split_response(text)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(Exception),
)
def markdown_proof_split_to_latex(
    client: OpenAI,
    model: str,
    markdown: str,
    max_tokens: int,
) -> Tuple[str, str]:
    """
    Convert a proof chunk and split it into:
      - proof environment part
      - trailing non-proof part
    """
    md_in = sanitize_ocr_markdown(markdown or "")
    md_ph, mapping, _seq = replace_display_math_with_placeholders(md_in)

    prompt = PROOF_SPLIT_PROMPT + md_ph.strip()
    raw = _chat_complete_text(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    raw = strip_code_fences(raw)
    raw = strip_outer_document(raw)
    _validate_llm_tex_output(raw)

    proof_part, rest_part = _split_proof_output(raw)
    if mapping:
        proof_part = restore_display_math_placeholders(proof_part, mapping)
        rest_part = restore_display_math_placeholders(rest_part, mapping)

    proof_tex = heal_latex_fragment(proof_part)
    rest_tex = heal_latex_fragment(rest_part)
    return proof_tex.strip(), rest_tex.strip()


_TAG_RE = re.compile(r"\\tag\*?\{([^}]+)\}")


def filter_and_dedupe_tags(latex: str, allowed_tags: set[str]) -> str:
    allowed = set([t.strip() for t in (allowed_tags or set()) if t and t.strip()])
    seen: set[str] = set()

    def _repl(m: re.Match) -> str:
        tag = (m.group(1) or "").strip()
        if allowed and tag not in allowed:
            return ""
        if tag in seen:
            return ""
        seen.add(tag)
        return m.group(0)

    return _TAG_RE.sub(_repl, latex or "")


def star_all_equation_like_envs(latex: str) -> str:
    """Suppress auto-numbering; we rely on recovered/manual tags."""
    s = latex or ""
    for base in ["equation", "align", "gather", "multline", "flalign"]:
        s = re.sub(rf"\\begin\{{{base}\}}", rf"\\begin{{{base}*}}", s)
        s = re.sub(rf"\\end\{{{base}\}}", rf"\\end{{{base}*}}", s)
    return s


# =========================
# Block sentinels (for downstream texTojson)
# =========================

ENV_BLOCK_RE = re.compile(
    r"\\begin\{(?P<env>defn|thm|lem|prop|cor|proof)\}\s*(?P<body>.*?)\\end\{\1\}",
    re.DOTALL,
)

NUMBERED_TITLE_RE = re.compile(
    r"\b(Theorem|Lemma|Proposition|Corollary|Definition)\s+([0-9]+(?:\.[0-9]+)*)\s*\.?",
    re.IGNORECASE,
)


def _first_nonempty_line(s: str) -> str:
    for ln in (s or "").splitlines():
        t = ln.strip()
        if t:
            return t
    return ""


def _strip_simple_latex_cmds(s: str) -> str:
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    return s.strip()


def extract_short_label(env: str, body: str) -> str:
    if env == "proof":
        return "Proof"

    head = "\n".join((body or "").splitlines()[:8])
    head_plain = _strip_simple_latex_cmds(head)

    m = NUMBERED_TITLE_RE.search(head_plain)
    if m:
        kind = m.group(1).title()
        num = m.group(2)
        return f"{kind} {num}"

    line = _strip_simple_latex_cmds(_first_nonempty_line(body))
    if not line:
        return "UNKNOWN"
    if len(line) > 120:
        return line[:120] + "..."
    return line


def escape_attr(s: str) -> str:
    s = (s or "").replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    return s


def insert_block_sentinels(latex: str) -> str:
    if "%<BLOCK" in (latex or ""):
        return (latex or "").strip()

    out: List[str] = []
    pos = 0
    src = latex or ""
    for m in ENV_BLOCK_RE.finditer(src):
        out.append(src[pos:m.start()])
        env = m.group("env")
        body = m.group("body")

        label = extract_short_label(env, body)
        out.append(f'%<BLOCK type={env} label="{escape_attr(label)}">\n')
        out.append(m.group(0))
        out.append("\n%</BLOCK>\n")
        pos = m.end()

    out.append(src[pos:])
    return "".join(out).strip()


def convert_blocks_to_latex(
    blocks: List[Block],
    client: OpenAI,
    model: str,
    max_tokens: int,
    workers: int = 4,
) -> List[str]:
    """Convert semantic blocks and preserve original order."""
    if not blocks:
        return []

    results: Dict[int, List[str]] = {}

    def _convert_one(idx: int, blk: Block) -> Tuple[int, List[str]]:
        try:
            if blk.kind == "heading":
                latex_h = heading_block_to_latex(blk.md)
                return idx, ([latex_h] if latex_h else [])

            if not blk.md.strip():
                return idx, []

            if blk.kind == "proof":
                latex = markdown_to_latex(client, model, blk.md, max_tokens=max_tokens)
                latex = insert_block_sentinels(latex)
                return idx, ([latex] if latex else [])

            latex = markdown_to_latex(client, model, blk.md, max_tokens=max_tokens)
            latex = insert_block_sentinels(latex)
            return idx, ([latex] if latex else [])

        except Exception as e:
            print(f"[convert] block {idx} ({blk.kind}) failed: {e}", file=sys.stderr)
            fallback = (blk.md or "").strip()
            if blk.kind == "heading" and fallback:
                fallback = heading_block_to_latex(fallback)
            return idx, ([fallback] if fallback else [])

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_convert_one, i, blk): i for i, blk in enumerate(blocks)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Semantic conversion (blocks)"):
            idx = futs[fut]
            try:
                i, outs = fut.result()
                results[i] = outs
            except Exception as e:
                print(f"[convert] block {idx} future failed: {e}", file=sys.stderr)
                fallback = (blocks[idx].md or "").strip()
                if blocks[idx].kind == "heading" and fallback:
                    fallback = heading_block_to_latex(fallback)
                results[idx] = [fallback] if fallback else []

    body_chunks: List[str] = []
    for i in range(len(blocks)):
        for piece in results.get(i, []):
            if piece and piece.strip():
                body_chunks.append(piece.strip())
    return body_chunks


# =========================
# Document template
# =========================

def build_tex_document(body_chunks: List[str]) -> str:
    preamble = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}

% OCR sometimes outputs non-standard optimization operators
\providecommand{\minimize}{\min}
\providecommand{\maximize}{\max}

% Theorem-like env names for downstream parsing
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{definition}[theorem]{Definition}

\begin{document}
"""
    end = r"""
\end{document}
"""
    preamble = preamble.replace("THEOREMSTYLE_MARKER", "\\theoremstyle{definition}")
    body = "\n\n".join([c for c in body_chunks if c and c.strip()]).rstrip()
    return preamble + body + "\n" + end


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

    # ---- Structural layer ----

    # (6) Attach standalone equation numbers like "(1.4)" back to preceding display math
    full_md = attach_standalone_equation_numbers(full_md)

    # (7) Inject heading sentinels to anchor structure
    anchored_md = inject_heading_sentinels(full_md)

    # (8) Greedy semantic chunking
    blocks = greedy_chunk_markdown(anchored_md)

    # (9) Split oversized para blocks to prevent LLM truncation
    max_chars = int(get_setting(settings, "MDTOTEX_MAX_CHARS", 3000))
    blocks = split_large_para_blocks(blocks, max_chars=max_chars)

    if not blocks:
        blocks = [Block(kind="para", env=None, md=full_md)]

    heading_count = sum(1 for b in blocks if b.kind == "heading")
    stmt_count    = sum(1 for b in blocks if b.kind == "stmt")
    proof_count   = sum(1 for b in blocks if b.kind == "proof")
    para_count    = sum(1 for b in blocks if b.kind == "para")
    print(
        f"[structure] blocks={len(blocks)} "
        f"(heading={heading_count}, stmt={stmt_count}, "
        f"proof={proof_count}, para={para_count})"
    )

    # ---- Conversion layer ----
    # (10) Convert semantic blocks in parallel while preserving order
    max_tokens = int(get_setting(settings, "MDTOTEX_MAX_TOKENS", 2048))
    body_chunks = convert_blocks_to_latex(
        blocks=blocks,
        client=client,
        model=model,
        max_tokens=max_tokens,
        workers=workers,
    )

    body_joined = "\n\n".join(body_chunks).strip()
    print(f"[convert] done — chunks={len(body_chunks)}, chars={len(body_joined)}")

    # ---- Post-fix / Assembly layer ----
    # (11) Final sanitation and deterministic assembly
    if not body_joined:
        print(f"ERROR: Conversion produced empty output", file=sys.stderr)
        sys.exit(3)

    allowed_tags = set(re.findall(r"\\tag\{([^}]+)\}", full_md))
    body_joined = filter_and_dedupe_tags(body_joined, allowed_tags)
    body_joined = heal_latex_fragment(body_joined)
    body_joined = star_all_equation_like_envs(body_joined)
    body_joined = heal_latex_fragment(body_joined)

    tex = build_tex_document([body_joined])

    # (12) Write output
    out_tex.write_text(tex, encoding="utf-8")
    print(f"[output] LaTeX written to {out_tex}")


if __name__ == "__main__":
    main()
