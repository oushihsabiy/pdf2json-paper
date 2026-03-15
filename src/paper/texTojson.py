#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert paper-style LaTeX (research papers) into example.json schema."""

from __future__ import annotations

import argparse
import json
import re
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from tqdm import tqdm

STREAM_REQUIRED: Optional[bool] = None


# -----------------------------
# User Fill Entry (edit here)
# -----------------------------
# Fill this with your desired source name.
# If non-empty, ALL output JSON items will use this value in field "source".
SOURCE_NAME_ENTRY = "OptimizationPaper"


# -----------------------------
# Config helpers
# -----------------------------

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
        raise KeyError(f"Missing/invalid '{key}' in config.json")
    return v.strip()


def get_cfg(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def find_settings_json() -> Path:
    path = Path(__file__).resolve().with_name("settings.json")
    if not path.exists():
        raise FileNotFoundError(f"settings.json not found next to script: {path}")
    return path


def load_settings() -> Dict[str, Any]:
    path = find_settings_json()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def get_setting(settings: Dict[str, Any], key: str, default: Any) -> Any:
    return settings.get(key, default)


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
        return (resp.choices[0].message.content or "")

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


# -----------------------------
# TeX stripping / cleaning
# -----------------------------

def strip_outer_document(tex: str) -> str:
    """Remove preamble and \\begin{document}/\\end{document} if present."""
    s = tex or ""
    s = re.sub(r"(?s).*?\\begin\{document\}\s*", "", s, count=1)
    s = re.sub(r"(?s)\\end\{document\}.*$", "", s, count=1)
    return s.strip()


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:json|latex)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


# -----------------------------
# Context parsing (ints only)
# -----------------------------

@dataclass
class Context:
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    chapter_number: int = 0
    section_number: int = 0
    subsection_number: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "chapter_number": int(self.chapter_number),
            "section_number": int(self.section_number),
            "subsection_number": int(self.subsection_number),
        }


# Heading commands (support starred)
CHAPTER_CMD_RE = re.compile(r"\\chapter\*?\{(.+?)\}")
SECTION_CMD_RE = re.compile(r"\\section\*?\{(.+?)\}")
SUBSECTION_CMD_RE = re.compile(r"\\subsection\*?\{(.+?)\}")
SUBSUBSECTION_CMD_RE = re.compile(r"\\subsubsection\*?\{(.+?)\}")

# Title parsing
_CHAPTER_TITLE_RE = re.compile(r"^\s*(?:CHAPTER|Chapter)\s+(\d+)\s*[:.\-–—]?\s*(.*)\s*$")
_SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:SECTION|Section|Sec\.?|EXERCISE|Exercise|Ex\.?|PROBLEM|Problem)\s+(\d+(?:\.\d+)*)\s*[:.\-–—]?\s*(.*)\s*$"
)
_CHAPTER_PREFIX_RE = re.compile(
    r"^\s*(?:CHAPTER|Chapter|Ch\.?)\s+(\d+(?:\.\d+)*)\s*[:.\-–—]?\s*(.*)\s*$"
)
_LEADING_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[:.\-–—]?\s*(.*)\s*$")


def _split_num_components(num: str) -> List[int]:
    out: List[int] = []
    for part in (num or "").strip().split("."):
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def _clean_heading_title_text(title: str) -> str:
    """
    Normalize heading text before number parsing.
    Keeps it conservative: remove only very common TeX wrappers/noise.
    """
    t = (title or "").strip()
    # unwrap common simple TeX emphasis macros
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\emph\{([^{}]*)\}", r"\1", t)
    # remove math wrappers if they wrap the whole title token
    t = t.replace(r"\(", "").replace(r"\)", "")
    t = t.replace(r"\[", "").replace(r"\]", "")
    # normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_chapter_title(title: str) -> Tuple[int, str]:
    """
    Returns (chapter_number:int, chapter_name:str). If unknown number -> 0.
    Accepts:
      - "Chapter 6: Large scale optimization"
      - "CHAPTER 6 Large scale optimization"
      - "6 Large scale optimization"  (fallback)
    """
    t = _clean_heading_title_text(title)
    m = _CHAPTER_TITLE_RE.match(t)
    if m:
        num = int(m.group(1))
        name = (m.group(2) or "").strip()
        return num, name

    m2 = _LEADING_NUM_RE.match(t)
    if m2:
        comps = _split_num_components(m2.group(1))
        if comps:
            return int(comps[0]), (m2.group(2) or "").strip()

    # no number
    return 0, t


def parse_section_like_title(title: str) -> Tuple[List[int], str]:
    """
    Parse section/exercise-like heading titles for optimization-paper style.

    Returns:
      - number components as ints, e.g. [1, 5] from "1.5 ..."
      - remaining title text (may be empty when heading is numeric-only)

    Common accepted forms in this PDF:
      - "1.5 Dual of intersection of cones"
      - "Section 1.5: Dual of intersection of cones"
      - "Exercise 1.5 Dual of intersection of cones"
      - "1.5.2 Some subsection"
      - "1.5" -> ([1, 5], "")
      - "A variant involving retained columns" -> ([], title)
    """
    t = _clean_heading_title_text(title)
    m = _SECTION_PREFIX_RE.match(t)
    if m:
        comps = _split_num_components(m.group(1))
        rest = (m.group(2) or "").strip()
        return comps, rest

    # Some sources place "Chapter 6.1 ..." in section-like headings.
    m_ch = _CHAPTER_PREFIX_RE.match(t)
    if m_ch:
        comps = _split_num_components(m_ch.group(1))
        rest = (m_ch.group(2) or "").strip()
        return comps, rest

    m2 = _LEADING_NUM_RE.match(t)
    if m2:
        comps = _split_num_components(m2.group(1))
        rest = (m2.group(2) or "").strip()
        # keep numeric-only headings too (e.g. "6.1"), caller can fallback title if needed.
        return comps, rest

    return [], t


# -----------------------------
# Node / unit building
# -----------------------------

PAGE_COMMENT_RE = re.compile(r"^\s*%+\s*={2,}.*Page\s+\d+.*={2,}\s*$", re.IGNORECASE)
CLEARPAGE_RE = re.compile(r"^\s*\\clearpage\s*$", re.IGNORECASE)

BLOCK_START_RE = re.compile(r'^\s*%<BLOCK\s+type=(\w+)\s+label="((?:\\.|[^"])*)">\s*$')
BLOCK_END_RE = re.compile(r'^\s*%</BLOCK>\s*$')

THEOREM_LIKE_BTYPE = {"defn", "thm", "lem", "prop", "cor", "alg", "algorithm"}
PROOF_BTYPE = {"proof", "solution"}

# Remove figures/tables early
_FIG_BEGIN_RE = re.compile(r"\\begin\{(figure|table)\*?\}")
_FIG_END_RE = re.compile(r"\\end\{(figure|table)\*?\}")

# Remove other redundant environments
_REDUNDANT_BEGIN_RE = re.compile(r"\\begin\{(abstract|acknowledgments?|bibliography|thebibliography|index|printindex)\*?\}")
_REDUNDANT_END_RE = re.compile(r"\\end\{(abstract|acknowledgments?|bibliography|thebibliography|index|printindex)\*?\}")


@dataclass
class Node:
    kind: str  # "text" | "block"
    ctx: Context
    text: str
    btype: Optional[str] = None
    blabel: Optional[str] = None


@dataclass
class Unit:
    kind: str  # "theorem_like" | "prose"
    ctx: Context
    latex: str
    hint_env: Optional[str] = None  # one of allowed env codes


def unescape_attr(s: str) -> str:
    # reverse of escape_attr used in mdTotex
    s = s.replace("\\n", "\n")
    s = s.replace('\\"', '"')
    s = s.replace("\\\\", "\\")
    return s


def build_nodes(tex_body: str) -> List[Node]:
    """
    Parse TeX body into nodes while tracking context.
    - Drops page scaffolding and \\clearpage
    - Drops figure/table environments entirely
    - Removes heading commands from the text stream (but updates context)
    - Drops \\section*{Contents} and its ToC body until next heading.
    - Extracts sentinel blocks as block nodes.
    """
    lines = (tex_body or "").splitlines()
    ctx = Context()
    nodes: List[Node] = []

    buf: List[str] = []
    buf_ctx: Optional[Context] = None

    def flush_buf() -> None:
        nonlocal buf, buf_ctx
        if not buf:
            return
        txt = "\n".join(buf).strip("\n")
        if txt.strip() and buf_ctx is not None:
            nodes.append(Node(kind="text", ctx=buf_ctx, text=txt))
        buf = []
        buf_ctx = None

    def set_buf_ctx_if_needed() -> None:
        nonlocal buf_ctx
        if buf_ctx is None:
            buf_ctx = Context(**ctx.__dict__)

    i = 0
    skip_contents = False
    fig_depth = 0
    redundant_depth = 0

    def _heading_matches(line: str) -> bool:
        return bool(
            CHAPTER_CMD_RE.search(line)
            or SECTION_CMD_RE.search(line)
            or SUBSECTION_CMD_RE.search(line)
            or SUBSUBSECTION_CMD_RE.search(line)
        )

    while i < len(lines):
        line = lines[i]

        # drop page scaffolding
        if PAGE_COMMENT_RE.match(line) or CLEARPAGE_RE.match(line):
            i += 1
            continue

        # drop figure/table environments
        if fig_depth == 0 and _FIG_BEGIN_RE.search(line):
            fig_depth = 1
            i += 1
            continue
        if fig_depth > 0:
            if _FIG_BEGIN_RE.search(line):
                fig_depth += 1
            if _FIG_END_RE.search(line):
                fig_depth -= 1
            i += 1
            continue

        # drop redundant environments (abstract, acknowledgments, bibliography, etc.)
        if redundant_depth == 0 and _REDUNDANT_BEGIN_RE.search(line):
            redundant_depth = 1
            i += 1
            continue
        if redundant_depth > 0:
            if _REDUNDANT_BEGIN_RE.search(line):
                redundant_depth += 1
            if _REDUNDANT_END_RE.search(line):
                redundant_depth -= 1
            i += 1
            continue

        # skip ToC after \section*{Contents}
        if skip_contents:
            if _heading_matches(line) or BLOCK_START_RE.match(line):
                skip_contents = False
                # do not increment, let loop re-handle this line as heading/block
                continue
            i += 1
            continue

        # sentinel block
        bs = BLOCK_START_RE.match(line)
        if bs:
            flush_buf()
            btype = (bs.group(1) or "").strip()
            blabel = unescape_attr((bs.group(2) or "").strip())
            j = i + 1
            block_lines: List[str] = []
            while j < len(lines) and not BLOCK_END_RE.match(lines[j]):
                block_lines.append(lines[j])
                j += 1
            block_text = "\n".join(block_lines).strip()
            nodes.append(Node(kind="block", ctx=Context(**ctx.__dict__), text=block_text, btype=btype, blabel=blabel))
            i = j + 1 if j < len(lines) else j
            continue

        # headings (update context, remove from stream)
        m = CHAPTER_CMD_RE.search(line)
        if m:
            flush_buf()
            chap_num, chap_name = parse_chapter_title(m.group(1))
            if chap_num:
                ctx.chapter_number = chap_num
            ctx.chapter = chap_name
            ctx.section = ""
            ctx.subsection = ""
            ctx.section_number = 0
            ctx.subsection_number = 0
            i += 1
            continue

        m = SECTION_CMD_RE.search(line)
        if m:
            flush_buf()
            title = m.group(1).strip()
            if title.strip().lower() == "contents":
                skip_contents = True
                i += 1
                continue

            comps, name = parse_section_like_title(title)
            # Typical: comps=[chapter, section]
            if len(comps) >= 2:
                if ctx.chapter_number == 0:
                    ctx.chapter_number = comps[0]
                ctx.section_number = comps[1]
                ctx.section = name if name else title
                ctx.subsection = ""
                ctx.subsection_number = 0
            else:
                # star section or unnumbered: treat as subsection title if already inside a section
                if ctx.section:
                    ctx.subsection = title
                    ctx.subsection_number = 0
                else:
                    # first section with no numbering
                    ctx.section = title
                    ctx.section_number = 0
                    ctx.subsection = ""
                    ctx.subsection_number = 0
            i += 1
            continue

        m = SUBSECTION_CMD_RE.search(line)
        if m:
            flush_buf()
            title = m.group(1).strip()
            comps, name = parse_section_like_title(title)
            if len(comps) >= 3:
                if ctx.chapter_number == 0:
                    ctx.chapter_number = comps[0]
                ctx.section_number = comps[1]
                ctx.subsection_number = comps[2]
                ctx.section = ctx.section or ""  # keep if already
                ctx.subsection = name if name else title
            else:
                ctx.subsection = title
                ctx.subsection_number = 0
            i += 1
            continue

        m = SUBSUBSECTION_CMD_RE.search(line)
        if m:
            flush_buf()
            title = m.group(1).strip()
            comps, name = parse_section_like_title(title)
            if len(comps) >= 3:
                if ctx.chapter_number == 0:
                    ctx.chapter_number = comps[0]
                ctx.section_number = comps[1]
                ctx.subsection_number = comps[2]
                ctx.subsection = name if name else title
            else:
                ctx.subsection = title
                ctx.subsection_number = 0
            i += 1
            continue

        # normal text line
        set_buf_ctx_if_needed()
        buf.append(line)
        i += 1

    flush_buf()
    return nodes


def build_units(nodes: List[Node], *, max_unit_chars: int) -> List[Unit]:
    """
    Convert nodes to LLM units.
    - theorem_like units: one statement block + optional following proof block
    - prose units: merge adjacent text nodes (same context) up to max_unit_chars,
      and also SPLIT any single oversized prose blob into multiple units (soft split)
      so that the LLM output stays within token limits.
    """
    units: List[Unit] = []
    i = 0

    def ctx_equal(a: Context, b: Context) -> bool:
        return a.__dict__ == b.__dict__

    def split_long_text(text: str, max_chars: int) -> List[str]:
        """Split a long prose blob into <= max_chars chunks (best-effort, paragraph-aware)."""
        s = (text or "").strip()
        if not s:
            return []
        if max_chars <= 0 or len(s) <= max_chars:
            return [s]

        # Split by blank lines first (paragraph boundaries), then greedy pack.
        paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        def flush() -> None:
            nonlocal cur, cur_len
            if cur:
                chunks.append("\n\n".join(cur).strip())
                cur = []
                cur_len = 0

        for p in paras:
            if not p:
                continue
            # If a single paragraph is too long, hard-split it by lines, then by chars.
            if len(p) > max_chars:
                flush()
                # try line-based
                lines = [ln.rstrip() for ln in p.splitlines() if ln.strip()]
                buf = ""
                for ln in lines:
                    cand = (buf + "\n" + ln).strip() if buf else ln
                    if len(cand) <= max_chars:
                        buf = cand
                    else:
                        if buf:
                            chunks.append(buf.strip())
                        # if line itself is still too long, hard cut
                        if len(ln) > max_chars:
                            for k in range(0, len(ln), max_chars):
                                chunks.append(ln[k:k+max_chars].strip())
                            buf = ""
                        else:
                            buf = ln
                if buf:
                    chunks.append(buf.strip())
                continue

            cand_len = cur_len + (2 if cur else 0) + len(p)
            if cand_len <= max_chars:
                cur.append(p)
                cur_len = cand_len
            else:
                flush()
                cur.append(p)
                cur_len = len(p)

        flush()

        # Safety: in case something slipped through, hard cut.
        final_chunks: List[str] = []
        for ch in chunks:
            if len(ch) <= max_chars:
                final_chunks.append(ch)
            else:
                for k in range(0, len(ch), max_chars):
                    final_chunks.append(ch[k:k+max_chars].strip())
        return [c for c in final_chunks if c]

    def emit_prose_units(ctx0: Context, merged: str) -> None:
        for part in split_long_text(merged, max_unit_chars):
            if part.strip():
                units.append(Unit(kind="prose", ctx=ctx0, latex=part.strip()))

    while i < len(nodes):
        nd = nodes[i]

        # theorem-like block
        if nd.kind == "block" and (nd.btype or "") in THEOREM_LIKE_BTYPE:
            stmt = (nd.text or "").strip()
            proof = ""
            if i + 1 < len(nodes):
                nd2 = nodes[i + 1]
                if nd2.kind == "block" and (nd2.btype or "") in PROOF_BTYPE:
                    proof = (nd2.text or "").strip()
                    i += 1  # consume proof
            latex = stmt
            if proof:
                latex = latex.rstrip() + "\n\n" + proof.lstrip()

            btype = (nd.btype or "").strip().lower()
            hint = {
                "defn": "def",
                "thm": "thm",
                "lem": "lem",
                "prop": "prop",
                "cor": "thm",
                "alg": "alg",
                "algorithm": "alg",
            }.get(btype, None)

            units.append(Unit(kind="theorem_like", ctx=nd.ctx, latex=latex, hint_env=hint))
            i += 1
            continue

        # proof blocks should have been consumed
        if nd.kind == "block" and (nd.btype or "") in PROOF_BTYPE:
            i += 1
            continue

        # prose node (merge by context, then split by size)
        if nd.kind == "text":
            ctx0 = nd.ctx
            parts = [nd.text.strip()]
            j = i + 1
            while j < len(nodes) and nodes[j].kind == "text" and ctx_equal(nodes[j].ctx, ctx0):
                cand = "\n\n".join(parts + [nodes[j].text.strip()])
                # NOTE: if the current chunk is already oversized, we stop merging here;
                # it will be split by emit_prose_units().
                if len(cand) > max_unit_chars and len("\n\n".join(parts)) >= max_unit_chars:
                    break
                if len(cand) > max_unit_chars:
                    break
                parts.append(nodes[j].text.strip())
                j += 1
            merged = "\n\n".join([p for p in parts if p]).strip()
            if merged:
                emit_prose_units(ctx0, merged)
            i = j
            continue

        i += 1

    return units
# LLM prompt + calls
# -----------------------------

def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Best-effort extraction of Retry-After from OpenAI-compatible error responses."""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(1.0, min(20.0, float(raw)))
    except Exception:
        return None


@retry(
    reraise=True,
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=5, max=20),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(APIError)
    ),
)
def llm_call(client: OpenAI, model: str, prompt: str, *, max_tokens: int) -> str:
    """Call the chat model and return text content."""
    try:
        text = _chat_complete_text(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
    except RateLimitError as e:
        retry_after = _retry_after_seconds(e)
        if retry_after is not None:
            time.sleep(retry_after)
        raise
    # Light client-side throttling to reduce unit-to-unit burst rate on strict gateways.
    time.sleep(1.0)
    return (text or "").strip()


def _default_cache_dir() -> Path:
    """A small on-disk cache to reduce token usage across re-runs."""
    # Prefer CWD so each book/project run keeps its own cache.
    d = Path.cwd() / ".cache_texTojson"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        # As a fallback (e.g., read-only CWD), use /tmp.
        d = Path("/tmp") / ".cache_texTojson"
        d.mkdir(parents=True, exist_ok=True)
    return d


def llm_call_cached(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> str:
    """Cache raw LLM responses keyed by (model, prompt)."""
    if not cache_enabled:
        return llm_call(client, model, prompt, max_tokens=max_tokens)

    cache_dir = cache_dir or _default_cache_dir()
    key = hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()
    path = cache_dir / f"{key}.txt"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            pass

    out = llm_call(client, model, prompt, max_tokens=max_tokens)
    try:
        path.write_text(out, encoding="utf-8")
    except Exception:
        pass
    return out


LATEX_TO_JSON_PROMPT_TEMPLATE = (
    "Convert ONE LaTeX snippet from an optimization research paper into structured JSON items.\n"
    "Return VALID JSON ONLY (no markdown, no comments, no extra text).\n\n"
    "Output: a JSON array. Each element is an object with keys:\n"
    "  - env: one of [\"def\",\"thm\",\"lem\",\"prop\",\"alg\"]\n"
    "  - content: the core statement in LaTeX (plain text or wrapped env is both acceptable)\n"
    "  - proof: either \"\" or the COMPLETE \\begin{proof}...\\end{proof} or \\begin{solution}...\\end{solution} from the LaTeX\n\n"
    "Rules (strict):\n"
    "1) Never output env outside the allowed list. Never output env='text'.\n"
    "2) Only extract theorem/lemma/proposition/algorithm/corollary when explicitly marked in the LaTeX (e.g., \\begin{theorem}, \\begin{lemma}, etc.). Skip implicit or prose statements.\n"
    "3) For definitions: extract both explicit (e.g., \\begin{definition}) and implicit definitions comprehensively.\n"
    "4) Preserve source fidelity: keep wording, formulas, numbering, qualifiers, and references.\n"
    "5) Do NOT hallucinate or complete missing math content.\n"
    "6) For research-paper text:\n"
    "   - Put the statement into content.\n"
    "   - Put the proof/solution text into proof only if explicitly present as \\begin{proof} or \\begin{solution}, otherwise set proof to \"\".\n"
    "7) Drop non-problem noise (ToC, page headers/footers, figure/table remnants, prompt leakage).\n"
    "8) If nothing usable as a theorem/definition/algorithm item exists, output [].\n\n"
    "Implicit dependency recovery mode: __IMPLICIT_MODE__\n"
    "- off: do not rewrite textual references.\n"
    "- llm: rewrite clear references like 'Theorem 6.1' -> 'Theorem~\\ref{thm:6.1}', 'Eq. (6.3)' -> '\\eqref{eq:6.3}'. If unsure, do not rewrite.\n\n"
    "LaTeX snippet:\n<<<LATEX>>>\n__LATEX__\n<<<END>>>\n"
)

EXTRACT_DEFINITIONS_PROMPT_TEMPLATE = (
    "Extract implicit definitions from the following LaTeX text from a research paper.\n"
    "Return VALID JSON ONLY (no markdown, no comments, no extra text).\n\n"
    "Output: a JSON array. Each element is an object with keys:\n"
    "  - content: the complete definition statement in LaTeX, ensuring logical coherence and clarity\n"
    "  - proof: \"\" (empty string, as definitions typically don't have proofs)\n\n"
    "Rules:\n"
    "1) Only extract definitions that are not already explicitly marked as such (e.g., not in \\begin{definition} environments).\n"
    "2) Look for implicit definitions indicated by phrases like 'we define', 'is defined as', 'let us denote', 'we say that', 'a ... is called', 'we introduce', 'denotes', etc.\n"
    "3) Ensure each extracted definition is self-contained, logically coherent, and includes all necessary mathematical notation and context.\n"
    "4) Preserve the original wording, formulas, and structure as much as possible.\n"
    "5) If a definition spans multiple sentences, combine them into a single coherent statement.\n"
    "6) Avoid extracting examples, theorems, or other non-definition content.\n"
    "7) If no implicit definitions are found, output [].\n\n"
    "LaTeX text:\n<<<LATEX>>>\n__LATEX__\n<<<END>>>\n"
)

# Robust JSON extraction: find the first top-level JSON value (array/object),
# respecting quoted strings and escapes (regex-based extraction breaks when LaTeX contains ']' etc.).
def _extract_first_json_value(text: str) -> str:
    s = strip_code_fences(text).strip()
    if not s:
        return ""

    # Fast path: whole string looks like JSON.
    if (s.startswith("[") and s.rstrip().endswith("]")) or (s.startswith("{") and s.rstrip().endswith("}")):
        return s.strip()

    # Find first '{' or '['
    start = None
    for i, ch in enumerate(s):
        if ch in "[{":
            start = i
            break
    if start is None:
        return ""

    stack: List[str] = []
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "[{":
            stack.append(ch)
            continue
        if ch in "]}":
            if not stack:
                continue
            top = stack[-1]
            if (top == "[" and ch == "]") or (top == "{" and ch == "}"):
                stack.pop()
                if not stack:
                    return s[start : j + 1].strip()
            else:
                # mismatched closer; ignore
                continue

    return ""


# -------- proof generation (for missing proofs) --------

def llm_extract_definitions_from_prose(
    client: OpenAI,
    model: str,
    latex: str,
    *,
    max_tokens: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Use LLM to extract implicit definitions from prose LaTeX text.
    Returns a list of dicts with 'content' and 'proof' (proof is always "").
    """
    latex = (latex or "").strip()
    if not latex:
        return []

    prompt = EXTRACT_DEFINITIONS_PROMPT_TEMPLATE.replace("__LATEX__", latex)

    raw = llm_call_cached(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    raw = strip_code_fences(raw)
    payload = _extract_first_json_value(raw)

    if not payload:
        return []

    try:
        data = json.loads(payload)
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict):
                    content = str(item.get("content", "")).strip()
                    if content:
                        out.append({"content": content, "proof": ""})
            return out
    except Exception:
        pass

    return []


def llm_latex_to_items(
    client: OpenAI,
    model: str,
    latex: str,
    *,
    implicit_mode: str,
    max_tokens: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
    repair_attempts: int = 1,
) -> List[Dict[str, Any]]:
    """
    Call LLM to convert ONE LaTeX snippet into a JSON array of items.

    Robustness goals:
    - tolerate the model returning a JSON object instead of an array
    - tolerate extra leading/trailing text by extracting the first top-level JSON value
    - one optional "repair" retry if JSON parsing fails
    """
    latex = (latex or "").strip()
    if not latex:
        return []

    prompt = LATEX_TO_JSON_PROMPT_TEMPLATE.replace("__IMPLICIT_MODE__", implicit_mode).replace("__LATEX__", latex)

    raw = llm_call_cached(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    raw = strip_code_fences(raw)
    payload = _extract_first_json_value(raw)

    def _coerce_items(obj: Any) -> List[Dict[str, Any]]:
        if obj is None:
            return []
        # If model returned {"items":[...]} or {"data":[...]} etc.
        if isinstance(obj, dict):
            for k in ["items", "data", "results"]:
                v = obj.get(k)
                if isinstance(v, list):
                    obj = v
                    break
            else:
                # single item dict
                obj = [obj]
        if isinstance(obj, list):
            out: List[Dict[str, Any]] = []
            for x in obj:
                if isinstance(x, dict):
                    out.append(x)
            return out
        return []

    tries = 0
    while True:
        tries += 1
        if not payload:
            return []

        try:
            data = json.loads(payload)
            return _coerce_items(data)
        except Exception:
            if tries > (1 + max(0, int(repair_attempts))):
                return []

            # Repair: ask the model to output valid JSON ONLY.
            repair_prompt = (
                "Your previous response was NOT valid JSON.\n"
                "Re-output VALID JSON ONLY (no markdown fences, no comments, no extra text).\n"
                "Return a JSON array of objects, each with keys: env, content, proof.\n"
                "If nothing is formalizable, return an empty array [].\n"
                "\n"
                "Original LaTeX snippet:\n"
                "<<<LATEX>>>\n"
                + latex
                + "\n<<<END>>>\n"
            )
            raw = llm_call_cached(
                client,
                model,
                repair_prompt,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
            )
            raw = strip_code_fences(raw)
            payload = _extract_first_json_value(raw)
    """
    Call LLM to convert ONE LaTeX snippet into a JSON array of items.

    Robustness goals:
    - tolerate the model returning a JSON object instead of an array
    - tolerate extra leading/trailing text by extracting the first top-level JSON value
    - one optional "repair" retry if JSON parsing fails
    """
    latex = (latex or "").strip()
    if not latex:
        return []

    prompt = LATEX_TO_JSON_PROMPT_TEMPLATE.replace("__IMPLICIT_MODE__", implicit_mode).replace("__LATEX__", latex)

    raw = llm_call_cached(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    raw = strip_code_fences(raw)
    payload = _extract_first_json_value(raw)

    def _coerce_items(obj: Any) -> List[Dict[str, Any]]:
        if obj is None:
            return []
        # If model returned {"items":[...]} or {"data":[...]} etc.
        if isinstance(obj, dict):
            for k in ["items", "data", "results"]:
                v = obj.get(k)
                if isinstance(v, list):
                    obj = v
                    break
            else:
                # single item dict
                obj = [obj]
        if isinstance(obj, list):
            out: List[Dict[str, Any]] = []
            for x in obj:
                if isinstance(x, dict):
                    out.append(x)
            return out
        return []

    tries = 0
    while True:
        tries += 1
        if not payload:
            return []

        try:
            data = json.loads(payload)
            return _coerce_items(data)
        except Exception:
            if tries > (1 + max(0, int(repair_attempts))):
                return []

            # Repair: ask the model to output valid JSON ONLY.
            repair_prompt = (
                "Your previous response was NOT valid JSON.\n"
                "Re-output VALID JSON ONLY (no markdown fences, no comments, no extra text).\n"
                "Return a JSON array of objects, each with keys: env, content, proof.\n"
                "If nothing is formalizable, return an empty array [].\n"
                "\n"
                "Original LaTeX snippet:\n"
                "<<<LATEX>>>\n"
                + latex
                + "\n<<<END>>>\n"
            )
            raw = llm_call_cached(
                client,
                model,
                repair_prompt,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
            )
            raw = strip_code_fences(raw)
            payload = _extract_first_json_value(raw)


# -------- statement/proof refinement (style normalization) --------

_NARRATIVE_CUES_RE = re.compile(
    r"(?i)\b(we|our|us|let\s+us|consider|recall|note\s+that|it\s+follows|it\s+should\s+be\s+apparent|as\s+discussed|in\s+this\s+section|we\s+now|we\s+have)\b"
)

_DEICTIC_METHOD_RE = re.compile(
    r"(?i)\b(the|this)\s+(method|procedure|algorithm)\s+(we\s+have\s+)?(just\s+)?(described|outlined|introduced)\b"
)


def statement_needs_refine(content: str, env_code: str) -> bool:
    """Cheap heuristic to decide whether a statement looks too narrative."""
    s = (content or "").strip()
    if not s:
        return False
    # Skip refinement for already-terse, math-heavy statements.
    if env_code in {"thm", "lem", "prop"} and len(s) < 400 and "\\" in s and not _NARRATIVE_CUES_RE.search(s):
        return False
    # Narrative cues.
    if _NARRATIVE_CUES_RE.search(s):
        return True
    # Definitions/algorithms are often narrative in papers.
    if env_code in {"def", "alg"}:
        # If it doesn't contain any math and is mostly prose, refine.
        if ("$" not in s) and ("\\[" not in s) and ("\\begin{" not in s or "\\begin{definition}" in s or "\\begin{algorithm}" in s):
            return True
    return False


def proof_needs_refine(proof: str) -> bool:
    s = (proof or "").strip()
    if not s:
        return False
    # Most proofs can keep narrative "We"; but for formalization we'd like to reduce.
    return bool(_NARRATIVE_CUES_RE.search(s))


REFINE_ITEM_PROMPT_TEMPLATE = (
    "You refine ONE extracted item from a LaTeX research paper.\n"
    "Return VALID JSON ONLY (no markdown fences, no comments, no extra text).\n\n"
    "Input JSON has keys env, content, proof. env is fixed and must remain the same.\n"
    "Allowed env values: def, thm, lem, prop, alg.\n\n"
    "Hard requirements:\n"
    "- Output JSON object with keys: env, content, proof.\n"
    "- content MUST be a COMPLETE LaTeX environment matching env (def->definition, thm->theorem, lem->lemma, prop->proposition, alg->algorithm).\n"
    "- proof MUST be either empty \"\" or a COMPLETE \\begin{proof}...\\end{proof}.\n"
    "- DO NOT add \\label or \\tag. (A later deterministic pass will insert labels.)\n"
    "- Preserve original notation, formulas, wording, and explicit references as much as possible.\n\n"
    "Style requirements (IMPORTANT):\n"
    "- Keep source fidelity first: preserve wording, notation, numbering, and references.\n"
    "- Only do minimal cleanup for malformed LaTeX / OCR glitches.\n"
    "- Do NOT rewrite just for style.\n"
    "- For research-paper extraction, prefer keeping source statement and source proof unchanged.\n\n"
    "Proof requirements:\n"
    "- If env is thm/lem/prop and proof is empty, you MUST generate a complete proof.\n"
    "- If proof is already present in the source, preserve its structure and references as much as possible; only lightly normalize if needed.\n"
    "- Do NOT add substantial new proof content when a complete source proof already exists.\n"
    "- When generating a missing proof, keep notation consistent and allow explicit references to earlier results if they are clearly required.\n"
    "- If unavoidable, add a comment line: %% ASSUMPTION: ...\n\n"
    "Recent earlier items (may help resolve references):\n"
    "__RECENT_ITEMS__\n\n"
    "Input item JSON:\n"
    "__ITEM_JSON__\n"
)


def llm_refine_item(
    client: OpenAI,
    model: str,
    *,
    env_code: str,
    content: str,
    proof: str,
    recent_items: List[str],
    max_tokens: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
    repair_attempts: int = 1,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM to rewrite ONE item into a more formal statement/proof."""
    env_code = (env_code or "").strip()
    if env_code not in ALLOWED_ENVS:
        return None

    item_obj = {"env": env_code, "content": (content or ""), "proof": (proof or "")}
    recent_txt = "\n".join([f"- {x}" for x in (recent_items or [])])
    if not recent_txt.strip():
        recent_txt = "- (none)"

    prompt = (
        REFINE_ITEM_PROMPT_TEMPLATE
        .replace("__RECENT_ITEMS__", recent_txt)
        .replace("__ITEM_JSON__", json.dumps(item_obj, ensure_ascii=False))
    )

    raw = llm_call_cached(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    raw = strip_code_fences(raw)
    payload = _extract_first_json_value(raw)
    if not payload:
        return None

    tries = 0
    while True:
        tries += 1
        try:
            data = json.loads(payload)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                data = data[0]
            if not isinstance(data, dict):
                return None
            return data
        except Exception:
            if tries > (1 + max(0, int(repair_attempts))):
                return None

            repair_prompt = (
                "Your previous response was NOT valid JSON.\n"
                "Re-output VALID JSON ONLY (no markdown fences, no comments, no extra text).\n"
                "Output ONE JSON object with keys: env, content, proof.\n"
                "env MUST stay as: " + env_code + "\n"
            )
            raw = llm_call_cached(
                client,
                model,
                repair_prompt + "\n" + prompt,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
            )
            raw = strip_code_fences(raw)
            payload = _extract_first_json_value(raw)


# Post-processing: env wrapping, labels, deps
# -----------------------------

ALLOWED_ENVS = {"def", "thm", "lem", "prop", "alg"}

_ENV_CANON = {
    "def": "def",
    "defn": "def",
    "definition": "def",
    "thm": "thm",
    "theorem": "thm",
    "lem": "lem",
    "lemma": "lem",
    "prop": "prop",
    "proposition": "prop",
    "cor": "thm",
    "corollary": "thm",
    "alg": "alg",
    "algorithm": "alg",
}

# Canonical LaTeX env names used inside content
_ENV_TO_LATEX_ENV = {
    "def": "definition",
    "thm": "theorem",
    "lem": "lemma",
    "prop": "proposition",
    "alg": "algorithm",
}

_ENV_KIND_NAME = {
    "def": "Definition",
    "thm": "Theorem",
    "lem": "Lemma",
    "prop": "Proposition",
    "alg": "Algorithm",
}

_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\(?:eqref|ref|autoref|pageref|cref|Cref)\{([^}]+)\}")

# Eq. (6.3), Equation (6.3), problem (6.3) etc.
_IMPLICIT_EQREF_RE = re.compile(
    r"(?<!\\)\b(?:Eq\.|Eqs\.|Equation|Equations|problem|constraint|model)\s*\(\s*(\d+(?:\.\d+)*)\s*\)",
    re.IGNORECASE,
)

# Bare "(6.3)" is ambiguous; only rewrite when preceded by "see" / "in" / "from" etc.
_IMPLICIT_PAREN_NUM_RE = re.compile(
    r"(?<!\\)\b(?:see|in|from|by|using|as\s+in)\s*\(\s*(\d+(?:\.\d+)*)\s*\)",
    re.IGNORECASE,
)

_TAG_NO_LABEL_RE = re.compile(r"\\tag\{([^}]+)\}(?!\s*\\label\{)")


def canon_env(env_raw: str, content: str = "") -> Optional[str]:
    """Canonicalize env to one of ALLOWED_ENVS; also infer from LaTeX begin env when needed."""
    e = (env_raw or "").strip().lower()
    e2 = _ENV_CANON.get(e)
    if e2 in ALLOWED_ENVS:
        return e2
    # infer from the first \begin{...} in content
    m = re.search(r"\\begin\{([A-Za-z]+)\}", content or "")
    if m:
        e3 = _ENV_CANON.get((m.group(1) or "").strip().lower())
        if e3 in ALLOWED_ENVS:
            return e3
    # weak fallback: infer from marker words inside content
    mm = re.search(r"\b(Definition|Theorem|Lemma|Proposition|Algorithm)\b", content or "", flags=re.IGNORECASE)
    if mm:
        k = mm.group(1).lower()
        e4 = _ENV_CANON.get(k)
        if e4 in ALLOWED_ENVS:
            return e4
    return None


def clean_llm_latex(s: str) -> str:
    s = strip_code_fences(s)
    # do NOT strip outer document here; content is usually env-only
    return (s or "").strip()


def _convert_begin_end_envnames(latex: str) -> str:
    """
    Convert short env names (thm/lem/prop/defn) to canonical long names
    inside begin/end tags.
    """
    if not latex:
        return latex
    repl = {
        "defn": "definition",
        "thm": "theorem",
        "lem": "lemma",
        "prop": "proposition",
        "cor": "corollary",
        "alg": "algorithm",
    }

    def _sub_begin(m: re.Match) -> str:
        env = m.group(1)
        env2 = repl.get(env.lower(), env)
        return f"\\begin{{{env2}}}"

    def _sub_end(m: re.Match) -> str:
        env = m.group(1)
        env2 = repl.get(env.lower(), env)
        return f"\\end{{{env2}}}"

    latex = re.sub(r"\\begin\{(\w+)\}", _sub_begin, latex)
    latex = re.sub(r"\\end\{(\w+)\}", _sub_end, latex)
    return latex


def ensure_wrapped_env(content: str, env_code: str) -> str:
    """
    Ensure `content` is wrapped in the correct *statement* environment.

    IMPORTANT: A plain snippet may contain inner environments (e.g., equation/align).
    Those do NOT count as an outer wrapper. We only treat the snippet as already-wrapped
    if it STARTS with a theorem-like environment (definition/theorem/lemma/proposition/algorithm).
    """
    content = (content or "").strip()
    content = _convert_begin_end_envnames(content)

    target_env = _ENV_TO_LATEX_ENV.get(env_code, "proposition")

    # If the snippet already starts with a theorem-like environment, normalize that outer env.
    m0 = re.match(r"(?s)^\s*\\begin\{([A-Za-z]+)\}", content)
    if m0:
        outer = (m0.group(1) or "").strip()
        outer_l = outer.lower()

        theorem_like = {"definition", "theorem", "lemma", "proposition", "algorithm", "corollary"}
        if outer_l in theorem_like:
            # Ensure it is closed (best-effort).
            end_pat = r"\\end\{" + re.escape(outer) + r"\}\s*$"
            if not re.search(end_pat, content):
                content = content.rstrip() + f"\n\\end{{{outer}}}"

            # Normalize outer env name to target_env if needed.
            if outer_l != target_env.lower():
                # IMPORTANT: use a callable repl so backslashes are not interpreted by re as escapes.
                content = re.sub(
                    r"^\s*\\begin\{" + re.escape(outer) + r"\}",
                    lambda _m: f"\\begin{{{target_env}}}",
                    content,
                    count=1,
                )
                content = re.sub(
                    r"\\end\{" + re.escape(outer) + r"\}\s*$",
                    lambda _m: f"\\end{{{target_env}}}",
                    content,
                    count=1,
                )
            return content.strip()

    # Otherwise (no wrapper, or starts with non-statement env like equation/align), wrap the whole snippet.
    return f"\\begin{{{target_env}}}\n{content}\n\\end{{{target_env}}}".strip()


def ensure_wrapped_proof(proof: str) -> str:
    proof = (proof or "").strip()
    proof = _convert_begin_end_envnames(proof)
    if not proof:
        return ""
    ms = re.search(r"(?is)\\begin\{solution\*?\}(.*?)\\end\{solution\*?\}", proof)
    if ms:
        inner = (ms.group(1) or "").strip()
        return f"\\begin{{proof}}\n{inner}\n\\end{{proof}}".strip()
    proof = re.sub(r"(?im)^\s*(?:\\textbf\{)?solution\s*[:.]?\s*(?:\})?\s*$", "", proof).strip()
    if not proof:
        return ""
    if r"\begin{proof}" in proof and r"\end{proof}" in proof:
        # keep first proof env only
        m = re.search(r"(?s)\\begin\{proof\}.*?\\end\{proof\}", proof)
        return (m.group(0).strip() if m else proof.strip())
    return f"\\begin{{proof}}\n{proof}\n\\end{{proof}}".strip()


def split_statement_and_solution(latex: str) -> Tuple[str, str]:
    """
    Split one snippet into (problem_statement, source_solution_text).
    For solution manuals:
    - Prefer explicit \\begin{solution}...\\end{solution}
    - Fallback to standalone "Solution." marker line.
    For research papers:
    - Prefer explicit \\begin{proof}...\\end{proof}
    """
    s = (latex or "").strip()
    if not s:
        return "", ""

    # For solution manuals
    m_env = re.search(r"(?is)\\begin\{solution\*?\}(.*?)\\end\{solution\*?\}", s)
    if m_env:
        sol = (m_env.group(1) or "").strip()
        stmt = (s[: m_env.start()] + "\n" + s[m_env.end() :]).strip()
        return stmt, sol

    m_line = re.search(r"(?im)^\s*(?:\\textbf\{)?solution\s*[:.]?\s*(?:\})?\s*$", s)
    if m_line:
        stmt = s[: m_line.start()].strip()
        sol = s[m_line.end() :].strip()
        return stmt, sol

    # For research papers: handle proof blocks
    m_proof = re.search(r"(?is)\\begin\{proof\}(.*?)\\end\{proof\}", s)
    if m_proof:
        proof = (m_proof.group(1) or "").strip()
        stmt = (s[: m_proof.start()] + "\n" + s[m_proof.end() :]).strip()
        return stmt, proof

    return s, ""


def insert_env_label(env_block: str, label_key: str) -> str:
    """
    Insert \\label{label_key} right after the \\begin{...} line if not present.
    """
    s = env_block.strip()
    if not s:
        return s
    if re.search(r"\\label\{" + re.escape(label_key) + r"\}", s):
        return s
    # if already has some label near top, keep it but still insert ours (needed for determinism)
    m = re.search(r"\\begin\{[A-Za-z]+\}", s)
    if not m:
        return s
    insert_pos = m.end()
    return (s[:insert_pos] + "\n\\label{" + label_key + "}\n" + s[insert_pos:].lstrip()).strip()


def insert_proof_label(proof_block: str, label_key: str) -> str:
    s = proof_block.strip()
    if not s:
        return s
    if re.search(r"\\label\{" + re.escape(label_key) + r"\}", s):
        return s
    m = re.search(r"\\begin\{proof\}", s)
    if not m:
        return s
    insert_pos = m.end()
    return (s[:insert_pos] + "\n\\label{" + label_key + "}\n" + s[insert_pos:].lstrip()).strip()


def extract_labels(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for m in _LABEL_RE.finditer(text):
        k = (m.group(1) or "").strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def extract_dependencies(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for m in _REF_RE.finditer(text):
        k = (m.group(1) or "").strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def apply_implicit_refs_rule(latex: str, *, known_label_keys: Optional[set[str]] = None) -> str:
    """
    Rewrite textual references into \\ref/\\eqref, but ONLY when the target label key already exists.

    This implements *implicit dependency recovery*:
      - "Theorem 6.1" -> "Theorem~\\ref{thm:6.1}" if "thm:6.1" is known.
      - "Eq. (6.3)"   -> "\\eqref{eq:6.3}" if "eq:6.3" is known.

    If known_label_keys is None, the function rewrites unconditionally (legacy behavior).
    """
    s = latex or ""
    known = known_label_keys

    def _has(key: str) -> bool:
        if known is None:
            return True
        return key in known

    # Handle chained kind references: "Theorems 6.1 and 6.2", "Definition 2.3, 2.4"
    kind_chain_re = re.compile(
        r"""
        (?<!\\)
        \b
        (?P<kind>Theorem|Lemma|Proposition|Definition|Algorithm|Corollary)
        s?\.?
        \s*
        (?:\(|\[)?\s*
        (?P<num>\d+(?:\.\d+)*)
        \s*(?:\)|\])?
        (?P<tail>
            (?:\s*(?:,|;|and|or|&)\s*(?:\(|\[)?\s*\d+(?:\.\d+)*\s*(?:\)|\])?)+
        )?
        (?P<punct>\.)?
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    num_only_re = re.compile(r"\d+(?:\.\d+)*")

    prefix_map = {
        "theorem": "thm",
        "lemma": "lem",
        "proposition": "prop",
        "definition": "def",
        "algorithm": "alg",
        "corollary": "thm",
    }

    def _repl_chain(m: re.Match) -> str:
        kind_raw = (m.group("kind") or "").strip()
        kind_l = kind_raw.lower()
        prefix = prefix_map.get(kind_l, "thm")

        num0 = (m.group("num") or "").strip()
        tail = m.group("tail") or ""
        punct = m.group("punct") or ""

        key0 = f"{prefix}:{num0}" if num0 else ""
        if num0 and _has(key0):
            head = f"{kind_raw}~\\ref{{{key0}}}"
        else:
            head = (f"{kind_raw} {num0}").rstrip()

        tail2 = ""
        if tail:
            def _tail_num_repl(mn: re.Match) -> str:
                numx = (mn.group(0) or "").strip()
                keyx = f"{prefix}:{numx}"
                if numx and _has(keyx):
                    return f"\\ref{{{keyx}}}"
                return numx

            tail2 = num_only_re.sub(_tail_num_repl, tail)

        return (head + tail2 + punct)

    s = kind_chain_re.sub(_repl_chain, s)

    # Eq. (6.3) / Equation (6.3) etc.
    def _eq_repl(m: re.Match) -> str:
        num = (m.group(1) or "").strip()
        key = f"eq:{num}"
        if num and _has(key):
            return f"\\eqref{{{key}}}"
        return m.group(0)

    s = _IMPLICIT_EQREF_RE.sub(_eq_repl, s)

    # "see (6.3)" -> \\eqref{eq:6.3}
    def _paren_repl(m: re.Match) -> str:
        num = (m.group(1) or "").strip()
        key = f"eq:{num}"
        if num and _has(key):
            return f"\\eqref{{{key}}}"
        return m.group(0)

    s = _IMPLICIT_PAREN_NUM_RE.sub(_paren_repl, s)

    return s

# -----------------------------
# Dependency preservation helpers
# -----------------------------

def dedup_preserve_order(xs):
    out = []
    seen = set()
    for x in xs or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_source_dep_keys(latex_snippet: str, *, known_label_keys: set[str]) -> list[str]:
    """Extract dependency label-keys from the *source* LaTeX snippet.

    Why: the LLM may rewrite a statement/proof to be more self-contained and accidentally drop
    textual cross-references like '(6.1)' or 'Theorem 6.2'. This function preserves those intended
    dependencies by extracting them BEFORE LLM rewriting.

    Strategy:
      - rewrite implicit textual refs to explicit \ref/\eqref using apply_implicit_refs_rule
        (only when the target label key is already known)
      - then extract the explicit referenced keys

    Only returns keys that already exist in known_label_keys (i.e., backward refs).
    """
    s = (latex_snippet or '').strip()
    if not s or not known_label_keys:
        return []

    # Preserve any explicit refs already in the snippet.
    # Also rewrite implicit references in the snippet into explicit ones (only for known keys).
    rewritten = apply_implicit_refs_rule(s, known_label_keys=known_label_keys)
    keys = extract_dependencies(rewritten)
    keys = [k for k in keys if k in known_label_keys]
    return dedup_preserve_order(keys)


def collect_ref_evidence(latex_snippet: str, dep_keys: list[str], *, max_chars: int = 180) -> dict[str, str]:
    """Collect a short evidence excerpt from the source snippet for each dep key."""
    s = latex_snippet or ''
    out: dict[str, str] = {}
    if not s:
        return out

    # small helpers
    def _clip(a: int, b: int) -> str:
        a = max(0, a)
        b = min(len(s), b)
        frag = s[a:b].strip()
        frag = re.sub(r"\s+", " ", frag)
        return frag

    prefix_to_kind = {
        'thm': 'Theorem',
        'lem': 'Lemma',
        'prop': 'Proposition',
        'def': 'Definition',
        'alg': 'Algorithm',
        'eq': 'Eq.',
    }

    for k in dep_keys or []:
        if k in out:
            continue

        # eq:6.1
        if k.startswith('eq:'):
            num = k.split(':', 1)[1]
            # try common textual forms
            pats = [
                re.compile(r"(?i)(Eq\.|Eqs\.|Equation|Equations|problem|constraint|model)\s*\(\s*" + re.escape(num) + r"\s*\)"),
                re.compile(r"\(\s*" + re.escape(num) + r"\s*\)"),
            ]
            m = None
            for p in pats:
                m = p.search(s)
                if m:
                    break
            if m:
                out[k] = _clip(m.start() - max_chars // 2, m.end() + max_chars // 2)
            continue

        # theorem-like keys: def:6.2, thm:6.1, ...
        m2 = re.match(r"^(thm|lem|prop|def|alg):(.+)$", k)
        if m2:
            pref = m2.group(1)
            num = m2.group(2)
            kind = prefix_to_kind.get(pref, 'Theorem')
            p = re.compile(r"(?i)\b" + re.escape(kind) + r"\s+" + re.escape(num) + r"\b")
            m = p.search(s)
            if m:
                out[k] = _clip(m.start() - max_chars // 2, m.end() + max_chars // 2)
            continue

    return out


REFS_PATCH_PROMPT_TEMPLATE = (
    "You are given a LaTeX statement environment (content) and an optional proof environment (proof).\n"
    "Your task is to ADD/RESTORE cross-references to earlier items/equations, using ONLY the provided label keys.\n\n"
    "Output format (STRICT): output VALID JSON ONLY with keys: content, proof.\n"
    "- content must remain a COMPLETE statement environment (definition/theorem/lemma/proposition/algorithm).\n"
    "- proof must be either empty \\\"\\\" or a COMPLETE \\begin{proof}...\\end{proof}.\n"
    "- Do NOT add \\label or \\tag. Do NOT remove any existing \\ref/\\eqref.\n"
    "- Make MINIMAL edits: only insert \\ref{...} or \\eqref{...} (possibly with short parentheticals like '(see ...)' ).\n"
    "- If a target is irrelevant to the current text, you may ignore it, but prefer to keep source-intended refs when possible.\n"
    "- You MAY add at most 2 additional references to the recent items list, but ONLY if the statement/proof clearly uses them.\n\n"
    "Targets to preserve (label keys + how to cite):\n"
    "__TARGETS__\n\n"
    "Evidence snippets from the source text (for placement):\n"
    "__EVIDENCE__\n\n"
    "Recent earlier items (optional; only cite if clearly needed):\n"
    "__RECENT__\n\n"
    "Current content:\n<<<CONTENT>>>\n__CONTENT__\n<<<END>>>\n\n"
    "Current proof:\n<<<PROOF>>>\n__PROOF__\n<<<END>>>\n"
)


TYPE_CLASSIFY_PROMPT_TEMPLATE = (
    "你是数学题分类器。请将下面题目严格分类为且仅为以下三类之一：\n"
    "1) 求值题\n"
    "2) 证明题\n"
    "3) 其他\n\n"
    "判定原则：\n"
    "- 要求证明、论证、验证某结论成立 -> 证明题\n"
    "- 要求计算、求解、求最优值、给出显式解、构造并求出数值/表达式 -> 求值题\n"
    "- 其余无法明确归入前两类 -> 其他\n\n"
    "只输出一个标签，不要输出其他内容。\n\n"
    "题目:\n<<<PROBLEM>>>\n__PROBLEM__\n<<<END>>>\n\n"
    "解答(可能为空):\n<<<PROOF>>>\n__PROOF__\n<<<END>>>\n"
)


DIRECT_ANSWER_PROMPT_TEMPLATE = (
    "你是数学解答抽取器。给定一道题目及其解答，请提取“最终答案”。\n"
    "仅当题目属于求值/计算/最优化求解类时提取；否则返回空字符串。\n\n"
    "输出要求（严格）：\n"
    "1) 只输出最终答案文本，不要解释，不要步骤，不要多余前后缀。\n"
    "2) 若无法确定唯一最终答案，输出空字符串。\n"
    "3) 若有明确最优值、闭式表达、最终数值或最终向量/矩阵，优先输出该结论。\n\n"
    "题目:\n<<<PROBLEM>>>\n__PROBLEM__\n<<<END>>>\n\n"
    "解答:\n<<<PROOF>>>\n__PROOF__\n<<<END>>>\n"
)


def _normalize_problem_type_label(s: str) -> Optional[str]:
    t = (s or "").strip()
    if t in {"求值题", "证明题", "其他"}:
        return t
    t2 = t.lower()
    if "prove" in t2 or "proof" in t2:
        return "证明题"
    if "compute" in t2 or "evaluate" in t2 or "calculate" in t2 or "solve" in t2:
        return "求值题"
    return None


def llm_classify_problem_type(
    client: OpenAI,
    model: str,
    *,
    problem: str,
    proof: str,
    max_tokens: int = 64,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> str:
    prompt = (
        TYPE_CLASSIFY_PROMPT_TEMPLATE
        .replace("__PROBLEM__", (problem or "").strip()[:6000])
        .replace("__PROOF__", (proof or "").strip()[:3000])
    )
    raw = llm_call_cached(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    raw2 = strip_code_fences(raw).strip()
    lab = _normalize_problem_type_label(raw2)
    if lab:
        return lab

    payload = _extract_first_json_value(raw2)
    if payload:
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict):
                lab2 = _normalize_problem_type_label(str(obj.get("type") or obj.get("label") or ""))
                if lab2:
                    return lab2
        except Exception:
            pass

    m = re.search(r"(求值题|证明题|其他)", raw2)
    if m:
        return m.group(1)
    return "其他"


def llm_extract_direct_answer(
    client: OpenAI,
    model: str,
    *,
    problem: str,
    proof: str,
    max_tokens: int = 128,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> str:
    prompt = (
        DIRECT_ANSWER_PROMPT_TEMPLATE
        .replace("__PROBLEM__", (problem or "").strip()[:6000])
        .replace("__PROOF__", (proof or "").strip()[:5000])
    )
    raw = llm_call_cached(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    s = strip_code_fences(raw).strip()
    # Keep output compact and single-line for JSON direct_answer.
    s = re.sub(r"\s+", " ", s).strip()
    if s in {"", '""', "''", "空", "无"}:
        return ""
    # Very long responses are likely non-compliant explanations; trim conservatively.
    if len(s) > 300:
        s = s[:300].strip()
    return s


def llm_patch_insert_refs(
    client: OpenAI,
    model: str,
    *,
    content: str,
    proof: str,
    required_keys: list[str],
    key_to_human: dict[str, str],
    evidence: dict[str, str],
    recent_items: list[str],
    max_tokens: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
    repair_attempts: int = 1,
) -> Optional[dict[str, str]]:
    """Ask LLM to re-insert missing refs with minimal edits."""
    req = dedup_preserve_order(required_keys)

    def _how(k: str) -> str:
        if k.startswith('eq:'):
            return f"\\eqref{{{k}}}"
        return f"\\ref{{{k}}}"

    if req:
        tgt_lines = []
        for k in req:
            human = key_to_human.get(k, '')
            if human:
                tgt_lines.append(f"- {k} ({human}): use {_how(k)}")
            else:
                tgt_lines.append(f"- {k}: use {_how(k)}")
        targets_txt = "\n".join(tgt_lines)
    else:
        targets_txt = "- (none)"

    if evidence:
        ev_lines = []
        for k, frag in evidence.items():
            if not frag:
                continue
            ev_lines.append(f"- {k}: {frag}")
        evidence_txt = "\n".join(ev_lines) if ev_lines else "- (none)"
    else:
        evidence_txt = "- (none)"

    recent_txt = "\n".join([f"- {x}" for x in (recent_items or [])])
    if not recent_txt.strip():
        recent_txt = "- (none)"

    prompt = (
        REFS_PATCH_PROMPT_TEMPLATE
        .replace('__TARGETS__', targets_txt)
        .replace('__EVIDENCE__', evidence_txt)
        .replace('__RECENT__', recent_txt)
        .replace('__CONTENT__', (content or '').strip())
        .replace('__PROOF__', (proof or '').strip())
    )

    raw = llm_call_cached(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    raw = strip_code_fences(raw)
    payload = _extract_first_json_value(raw)
    if not payload:
        return None

    tries = 0
    while True:
        tries += 1
        try:
            data = json.loads(payload)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                data = data[0]
            if not isinstance(data, dict):
                return None
            c2 = str(data.get('content', '') or '').strip()
            p2 = str(data.get('proof', '') or '').strip()
            return {'content': c2, 'proof': p2}
        except Exception:
            if tries > (1 + max(0, int(repair_attempts))):
                return None
            repair_prompt = (
                "Your previous response was NOT valid JSON.\n"
                "Re-output VALID JSON ONLY with keys: content, proof.\n"
                "No markdown fences, no comments.\n"
            )
            raw = llm_call_cached(
                client,
                model,
                repair_prompt + "\n" + prompt,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
            )
            raw = strip_code_fences(raw)
            payload = _extract_first_json_value(raw)


def label_tagged_equations(latex: str) -> str:
    """
    For every \\tag{X} that is not immediately followed by a \\label{...},
    inject \\label{eq:X}.
    """
    def _repl(m: re.Match) -> str:
        num = (m.group(1) or "").strip()
        # sanitize
        num = re.sub(r"\s+", "", num)
        return f"\\tag{{{num}}}\\label{{eq:{num}}}"
    return _TAG_NO_LABEL_RE.sub(_repl, latex or "")


_MATH_ENV_RE = re.compile(
    r"(?s)"
    r"\\begin\{(?P<env>equation\*?|gather\*?|multline\*?|align\*?|flalign\*?|alignat\*?)\}"
    r"(?P<body>.*?)"
    r"\\end\{(?P=env)\}"
)

_BRACKET_MATH_RE = re.compile(r"(?s)\\\[(?P<body>.*?)\\\]")


def inject_missing_eq_labels(
    latex: str,
    *,
    chapter_number: int,
    used_labels: Optional[set] = None,
    start_index: int = 1,
) -> Tuple[str, int]:
    """
    Add \\label{...} to display math blocks that still have no label after tag-based injection.
    Label scheme (deterministic, collision-avoiding):
      eq:<chapter>.u<k>
    """
    if not latex:
        return latex, start_index
    used = used_labels if used_labels is not None else set()
    k = start_index

    def _new_label() -> str:
        nonlocal k
        while True:
            key = f"eq:{chapter_number}.u{k}"
            k += 1
            if key not in used:
                used.add(key)
                return key

    # equation-like envs
    parts: List[str] = []
    last = 0
    for m in _MATH_ENV_RE.finditer(latex):
        parts.append(latex[last:m.start()])
        env = m.group("env")
        body = m.group("body") or ""
        block = m.group(0)

        if "\\label{" in block:
            parts.append(block)
        else:
            lab = _new_label()
            # insert label right after begin
            block2 = re.sub(r"(\\begin\{" + re.escape(env) + r"\})", r"\1\n\\label{" + lab + r"}", block, count=1)
            parts.append(block2)
        last = m.end()
    parts.append(latex[last:])
    latex2 = "".join(parts)

    # \[ ... \] blocks (only if they don't have any label)
    parts = []
    last = 0
    for m in _BRACKET_MATH_RE.finditer(latex2):
        parts.append(latex2[last:m.start()])
        body = m.group("body") or ""
        block = m.group(0)
        if "\\label{" in block:
            parts.append(block)
        else:
            lab = _new_label()
            # insert label right after \[
            parts.append("\\[\n\\label{" + lab + "}\n" + body.strip() + "\n\\]")
        last = m.end()
    parts.append(latex2[last:])
    return "".join(parts), k


def looks_non_informative(content: str) -> bool:
    """
    Conservative safety filter (after LLM):
    drop only clearly non-statement / non-formalizable artifacts.

    IMPORTANT: do NOT drop items merely because they are short; valid definitions
    and claims can be brief.
    """
    s = (content or "").strip()
    if not s:
        return True

    # Table of contents artifacts / headings
    if re.search(r"\\section\*?\{Contents\}", s, flags=re.IGNORECASE):
        return True
    if re.search(r"^\s*\\section\*?\{", s) or re.search(r"^\s*\\chapter\*?\{", s):
        return True
    if "Table of Contents" in s:
        return True

    # Figure/table remnants
    if "\\includegraphics" in s:
        return True
    if re.search(r"\\begin\{figure\*?\}", s, flags=re.IGNORECASE):
        return True
    if re.search(r"\\begin\{table\*?\}", s, flags=re.IGNORECASE):
        return True

    return False


def convert_tex_to_items(
    tex: str,
    *,
    client: OpenAI,
    json_model: str,
    max_tokens_json: int,
    implicit_mode: str = "rule",   # off|rule|llm
    max_unit_chars: int = 6000,
    # optional refinement pass (mostly disabled in source-first solution-manual mode)
    refine_mode: str = "auto",     # off|auto|all
    refine_model: Optional[str] = None,
    max_tokens_refine: int = 2048,
    # new: response cache (token saving across re-runs)
    cache_enabled: bool = True,
    cache_dir: Optional[Path] = None,
    # new: dependency preservation / review
    dep_review_mode: str = "off",  # off|llm
    dep_patch_model: Optional[str] = None,
    max_tokens_dep_patch: int = 1024,
) -> List[Dict[str, Any]]:
    """
    End-to-end extraction for solution-manual style books:
      - parse nodes -> build units
      - LLM: unit -> tentative items [{env, content}]
      - post-process: canonical env, wrapping, labels/deps/context bookkeeping
      - final mapping is done by `to_example_output_schema` (problem/proof/type/source/source_idx)
    """
    body = strip_outer_document(tex)
    refine_model = (refine_model or json_model)
    dep_patch_model = (dep_patch_model or json_model)
    if cache_enabled and cache_dir is None:
        cache_dir = _default_cache_dir()
    nodes = build_nodes(body)
    units = build_units(nodes, max_unit_chars=max_unit_chars)

    # deterministic counters per chapter and per env
    env_counts: Dict[int, Dict[str, int]] = {}
    eq_auto_next: Dict[int, int] = {}  # chapter -> next u-index
    used_label_keys: Dict[int, set] = {}  # chapter -> set of label keys used (eq/def/thm/...)

    items_out: List[Dict[str, Any]] = []

    # Map LaTeX \label keys -> earlier item 'label' (for dependency resolution)
    known_label_to_item: Dict[str, str] = {}

    # For resolving deictic phrases like "the method we just described".
    # Keyed by (chapter_number, section_number, subsection_number).
    last_alg_label_key: Dict[Tuple[int, int, int], str] = {}

    def _ctx_key(c: Context) -> Tuple[int, int, int]:
        return (int(c.chapter_number or 0), int(c.section_number or 0), int(c.subsection_number or 0))

    def _recent_item_labels(c: Context, limit: int = 8) -> List[str]:
        """Recent item human labels within the same local context (for LLM refinement hints)."""
        key = _ctx_key(c)
        out: List[str] = []
        for it in reversed(items_out):
            ctxj = it.get("context") or {}
            keyj = (
                int(ctxj.get("chapter_number") or 0),
                int(ctxj.get("section_number") or 0),
                int(ctxj.get("subsection_number") or 0),
            )
            if keyj == key:
                out.append(str(it.get("label") or "").strip())
            if len(out) >= limit:
                break
        return list(reversed([x for x in out if x]))

    def _apply_deictic_method_refs(s: str, c: Context) -> str:
        """Replace 'the method we just described' with an explicit ref to the most recent algorithm."""
        if not s:
            return s
        key = _ctx_key(c)
        alg_key = last_alg_label_key.get(key)
        if not alg_key:
            return s
        return _DEICTIC_METHOD_RE.sub(r"the method described in Algorithm~\\ref{" + alg_key + r"}", s)

    for u in tqdm(units, desc="LLM units", unit="unit"):
        ctx = u.ctx
        chap = int(ctx.chapter_number or 0)

        # Preserve dependency hints from the SOURCE snippet (LLM may drop references when rewriting).
        known_keys_before = set(known_label_to_item.keys())
        unit_src_dep_keys = extract_source_dep_keys(u.latex or '', known_label_keys=known_keys_before)
        unit_ref_evidence = collect_ref_evidence(u.latex or '', unit_src_dep_keys)

        # Source fallback: keep original statement/proof/solution from this unit.
        orig_stmt_txt, orig_proof_txt = split_statement_and_solution(u.latex or "")
        mp0 = re.search(r"(?s)\\begin\{proof\}.*?\\end\{proof\}", orig_stmt_txt or "")
        if mp0:
            if not orig_proof_txt.strip():
                orig_proof_txt = (orig_stmt_txt[mp0.start() : mp0.end()] or "").strip()
            orig_stmt_txt = ((orig_stmt_txt[: mp0.start()] or "") + "\n" + (orig_stmt_txt[mp0.end() :] or "")).strip()

        raw_items = llm_latex_to_items(
            client=client,
            model=json_model,
            latex=u.latex,
            implicit_mode=("llm" if implicit_mode == "llm" else "off"),
            max_tokens=max_tokens_json,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )

        # If LLM returns nothing:
        # - for theorem_like units, keep the original statement/proof as a fallback
        # - for prose units, try to extract implicit definitions
        if not raw_items:
            if u.kind == "theorem_like":
                fallback_env = u.hint_env or "prop"
                raw_items = [
                    {
                        "env": fallback_env,
                        "content": orig_stmt_txt or (u.latex or "").strip(),
                        "proof": orig_proof_txt,
                    }
                ]
            else:
                # Try to extract implicit definitions from prose
                try:
                    definitions = llm_extract_definitions_from_prose(
                        client=client,
                        model=json_model,
                        latex=u.latex,
                        max_tokens=max_tokens_json,
                        cache_dir=cache_dir,
                        cache_enabled=cache_enabled,
                    )
                    if definitions:
                        raw_items = [{"env": "def", **d} for d in definitions]
                    else:
                        # Prose units from solution manuals may still contain a Q/S pair.
                        if orig_stmt_txt.strip() and orig_proof_txt.strip():
                            raw_items = [
                                {
                                    "env": "prop",
                                    "content": orig_stmt_txt.strip(),
                                    "proof": orig_proof_txt.strip(),
                                }
                            ]
                        else:
                            continue
                except NameError:
                    # Fallback if function not defined
                    if orig_stmt_txt.strip() and orig_proof_txt.strip():
                        raw_items = [
                            {
                                "env": "prop",
                                "content": orig_stmt_txt.strip(),
                                "proof": orig_proof_txt.strip(),
                            }
                        ]
                    else:
                        continue

        # For theorem-like snippets, enforce ONE item.
        if u.kind == "theorem_like" and len(raw_items) > 1:
            raw_items = raw_items[:1]

        # Source-level deps are only safe to attribute when the unit yields ONE item (theorem-like, or single extraction).
        use_unit_src_deps = (u.kind == "theorem_like" or len(raw_items) == 1)

        for rit in raw_items:
            env0 = canon_env(str(rit.get("env", "")), str(rit.get("content", "") or ""))
            if env0 is None:
                # For theorem-like input, fall back to sentinel hint env if LLM gave an invalid env.
                if u.kind == "theorem_like" and (u.hint_env or "") in ALLOWED_ENVS:
                    env0 = u.hint_env
                else:
                    continue

            content = clean_llm_latex(str(rit.get("content", "") or "")).strip()
            proof = clean_llm_latex(str(rit.get("proof", "") or "")).strip()
            # For paper-style extraction, allow LLM to extract both explicit and implicit proofs from source text
            # proof = ensure_wrapped_proof(orig_proof_txt.strip()) if orig_proof_txt.strip() else ""
            lock_source_proof = True

            # For theorem-like units, prefer the env hint from sentinels (more reliable than LLM).
            if u.kind == "theorem_like" and (u.hint_env or "") in ALLOWED_ENVS:
                env0 = u.hint_env  # type: ignore[assignment]

            # If the model dropped statement/proof on theorem-like input, fall back to originals.
            if u.kind == "theorem_like":
                if (not content.strip()) and orig_stmt_txt.strip():
                    content = orig_stmt_txt.strip()
                if (not proof.strip()) and orig_proof_txt.strip():
                    proof = orig_proof_txt.strip()

            if (not content.strip()) and orig_stmt_txt.strip():
                content = orig_stmt_txt.strip()

            # Filter only for prose-derived items (never drop theorem-like items).
            if u.kind != "theorem_like" and looks_non_informative(content):
                continue

            # ensure wrappers
            content = ensure_wrapped_env(content, env0)
            proof = ensure_wrapped_proof(proof)

            # Optional refinement pass:
            # In this project we use source-first extraction (solution text is trusted),
            # so refinement is typically skipped when source proof is locked.
            do_refine = False
            if lock_source_proof:
                do_refine = False
            elif refine_mode == "all":
                do_refine = True
            elif refine_mode == "auto":
                if u.kind == "theorem_like":
                    do_refine = True
                elif statement_needs_refine(content, env0) or proof_needs_refine(proof):
                    do_refine = True

            if do_refine and refine_model:
                recent = _recent_item_labels(ctx, limit=8)
                refined = llm_refine_item(
                    client=client,
                    model=refine_model,
                    env_code=env0,
                    content=content,
                    proof=proof,
                    recent_items=recent,
                    max_tokens=max_tokens_refine,
                    cache_dir=cache_dir,
                    cache_enabled=cache_enabled,
                )
                if isinstance(refined, dict):
                    env_r = canon_env(str(refined.get("env", "")), str(refined.get("content", "") or "")) or env0
                    # For theorem-like units, trust the sentinel env hint.
                    if u.kind == "theorem_like" and (u.hint_env or "") in ALLOWED_ENVS:
                        env_r = u.hint_env  # type: ignore[assignment]

                    content_r = clean_llm_latex(str(refined.get("content", "") or "")).strip()
                    proof_r = clean_llm_latex(str(refined.get("proof", "") or "")).strip()

                    # If refinement accidentally dropped important content, keep the previous version.
                    if content_r.strip():
                        content = ensure_wrapped_env(content_r, env_r)
                    else:
                        content = ensure_wrapped_env(content, env0)

                    if proof_r.strip():
                        proof = ensure_wrapped_proof(proof_r)
                    else:
                        proof = ensure_wrapped_proof(proof)
                    env0 = env_r

            # optional rule-based implicit refs
            if implicit_mode == "rule":
                known_keys = set(known_label_to_item.keys())
                content = apply_implicit_refs_rule(content, known_label_keys=known_keys)
                if proof and (not lock_source_proof):
                    proof = apply_implicit_refs_rule(proof, known_label_keys=known_keys)

                # Resolve deictic phrases like "the method we just described" to the most recent
                # algorithm in the same local context (if any).
                content = _apply_deictic_method_refs(content, ctx)
                if proof and (not lock_source_proof):
                    proof = _apply_deictic_method_refs(proof, ctx)


            # -------------------------------------------------
            # Dependency preservation:
            # - extract deps from final content/proof (explicit \ref/\eqref)
            # - ALSO carry deps detected in the SOURCE snippet (unit_src_dep_keys)
            #   because LLM rewriting/refinement may delete reference cues.
            #
            # For multi-item prose units, unit_src_dep_keys is NOT attributed (too ambiguous).
            # -------------------------------------------------
            src_dep_keys_for_item = unit_src_dep_keys if use_unit_src_deps else []
            # Keep only backward refs (already-known label keys).
            src_dep_keys_for_item = [k for k in src_dep_keys_for_item if k in known_keys_before]

            # Optional LLM review to (a) re-insert missing refs from source and/or
            # (b) add a small number of obvious refs to recent items.
            if dep_review_mode == "llm" and dep_patch_model and use_unit_src_deps and (not lock_source_proof):
                cur_ref_keys = set(extract_dependencies(content) + extract_dependencies(proof))
                missing_keys = [k for k in src_dep_keys_for_item if k not in cur_ref_keys]

                # Cap to keep prompts small (token saving)
                if len(missing_keys) > 12:
                    missing_keys = missing_keys[:12]


                # Token-aware: only review theorem/lemma/proposition items, or when source refs were dropped.
                review_needed = bool(missing_keys)
                if (not review_needed) and (not cur_ref_keys) and (env0 in PROOF_REQUIRED_ENVS) and (len(content) > 800):
                    review_needed = True

                if review_needed:
                    # recent items in the same local context, shown with their label keys
                    recent_refs: list[str] = []
                    key0 = _ctx_key(ctx)
                    for it_prev in reversed(items_out):
                        ctxj = it_prev.get("context") or {}
                        keyj = (
                            int(ctxj.get("chapter_number") or 0),
                            int(ctxj.get("section_number") or 0),
                            int(ctxj.get("subsection_number") or 0),
                        )
                        if keyj != key0:
                            continue
                        nc = it_prev.get("number_components") or []
                        envp = str(it_prev.get("env") or "").strip()
                        if envp in ALLOWED_ENVS and isinstance(nc, list) and len(nc) >= 2:
                            kkey = f"{envp}:{int(nc[0])}.{int(nc[1])}"
                            recent_refs.append(f"{it_prev.get('label','')} => {kkey}")
                        if len(recent_refs) >= 10:
                            break
                    recent_refs = list(reversed([x for x in recent_refs if x.strip()]))

                    key_to_human = {k: known_label_to_item.get(k, "") for k in (missing_keys or [])}
                    evidence_subset = {k: unit_ref_evidence.get(k, "") for k in (missing_keys or []) if unit_ref_evidence.get(k, "")}
                    patched = llm_patch_insert_refs(
                        client=client,
                        model=dep_patch_model,
                        content=content,
                        proof=proof,
                        required_keys=missing_keys,
                        key_to_human=key_to_human,
                        evidence=evidence_subset,
                        recent_items=recent_refs,
                        max_tokens=max_tokens_dep_patch,
                        cache_dir=cache_dir,
                        cache_enabled=cache_enabled,
                    )
                    if isinstance(patched, dict) and patched.get('content'):
                        content = ensure_wrapped_env(clean_llm_latex(patched.get('content','')), env0)
                        proof = ensure_wrapped_proof(clean_llm_latex(patched.get('proof','')))
                        # Re-apply implicit ref rewriting in case the model reintroduced textual cues.
                        if implicit_mode == "rule":
                            known_keys = set(known_label_to_item.keys())
                            content = apply_implicit_refs_rule(content, known_label_keys=known_keys)
                            if proof:
                                proof = apply_implicit_refs_rule(proof, known_label_keys=known_keys)

            # (Dependencies are computed later after label injection.)

            # assign deterministic label (chapter + per-env count)
            env_counts.setdefault(chap, {})
            env_counts[chap].setdefault(env0, 0)
            env_counts[chap][env0] += 1
            k = env_counts[chap][env0]

            kind_name = _ENV_KIND_NAME[env0]
            label = f"{kind_name} {chap}.{k}"
            number_components = [chap, k]

            # label keys used inside latex
            used_label_keys.setdefault(chap, set())
            used = used_label_keys[chap]

            item_label_key = f"{env0}:{chap}.{k}"  # e.g., thm:6.1
            proof_label_key = f"prf:{env0}:{chap}.{k}"

            # Update most-recent algorithm key for deictic reference resolution.
            if env0 == "alg":
                last_alg_label_key[_ctx_key(ctx)] = item_label_key

            content = insert_env_label(content, item_label_key)
            if proof and (not lock_source_proof):
                proof = insert_proof_label(proof, proof_label_key)

            # equation label injection
            content = label_tagged_equations(content)
            if proof and (not lock_source_proof):
                proof = label_tagged_equations(proof)

            # collect already-used labels so we don't collide when injecting missing ones
            for key in extract_labels(content) + extract_labels(proof):
                used.add(key)

            eq_start = eq_auto_next.get(chap, 1)
            content, eq_next = inject_missing_eq_labels(content, chapter_number=chap, used_labels=used, start_index=eq_start)
            eq_auto_next[chap] = eq_next
            if proof and (not lock_source_proof):
                proof, eq_next2 = inject_missing_eq_labels(proof, chapter_number=chap, used_labels=used, start_index=eq_auto_next.get(chap, eq_next))
                eq_auto_next[chap] = eq_next2

            # recompute extracted_labels (all \label keys in content+proof)
            extracted = extract_labels(content) + extract_labels(proof)
            # de-dup preserve order
            seen = set()
            extracted2: List[str] = []
            for x in extracted:
                if x not in seen:
                    seen.add(x)
                    extracted2.append(x)

            # dependencies:
            # - explicit deps: referenced label keys that already appeared in previous items' extracted_labels
            # - implicit deps (rule mode): handled above by rewriting textual refs into \ref/\eqref
            ref_keys = extract_dependencies(content) + extract_dependencies(proof) + list(src_dep_keys_for_item)
            deps_item_labels: List[str] = []
            seen_dep: set[str] = set()
            for rk in ref_keys:
                dep_item = known_label_to_item.get(rk)
                if dep_item and dep_item != label and dep_item not in seen_dep:
                    seen_dep.add(dep_item)
                    deps_item_labels.append(dep_item)

            cur_pos = len(items_out) + 1  # 1-based order in document
            items_out.append(
                {
                    "label": label,
                    "env": env0,
                    "number_components": number_components,
                    "extracted_labels": extracted2,
                    "context": ctx.to_json(),
                    "content": content.strip(),
                    "dependencies": deps_item_labels,
                    "proof": proof.strip(),
                    "index": 0,  # fill later
                }
            )

            # Update known label map AFTER appending (so only previous items are deps for future items)
            for lk in extracted2:
                if lk not in known_label_to_item:
                    known_label_to_item[lk] = label

    # assign indices
    for idx, it in enumerate(items_out, start=1):
        it["index"] = idx

    # Trim internal fields not needed by final JSON mapping.
    trimmed: List[Dict[str, Any]] = []
    for it in items_out:
        trimmed.append(
            {
                "index": int(it.get("index", 0) or 0),
                "content": str(it.get("content") or ""),
                "proof": str(it.get("proof") or ""),
            }
        )
    return trimmed


def to_example_output_schema(
    items: List[Dict[str, Any]],
    *,
    source_fixed: Optional[str] = None,
    type_client: Optional[OpenAI] = None,
    type_model: Optional[str] = None,
    type_max_tokens: int = 64,
    type_cache_enabled: bool = True,
    type_cache_dir: Optional[Path] = None,
    answer_client: Optional[OpenAI] = None,
    answer_model: Optional[str] = None,
    answer_max_tokens: int = 128,
    answer_cache_enabled: bool = True,
    answer_cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Map internal items to final `example.json` fields:
    index, problem, proof, direct_answer, 题目类型, 预估难度, source, source_idx.
    Tailored for optimization-paper PDFs.
    """
    def _marker_pattern() -> re.Pattern:
        # Supports:
        #   (a) / (A)
        #   a) / A)
        #   (i) / (ii) / (iv)
        return re.compile(r"(?m)^\s*(?:\(([a-zA-Z])\)|([a-zA-Z])\)|\(([ivxIVX]+)\))\s+")

    def _normalize_part(m: re.Match) -> str:
        g1 = m.group(1) or ""
        g2 = m.group(2) or ""
        g3 = m.group(3) or ""
        if g1:
            return g1.lower()
        if g2:
            return g2.lower()
        if g3:
            return g3.lower()
        return ""

    def extract_problem_number(problem: str, fallback_chap: int, fallback_ex: int) -> Tuple[int, int]:
        """
        Prefer original numbering from the problem text:
          2.13 ...
          Exercise 2.13 ...
        Fallback to provided chapter/ex counters.
        """
        s = (problem or "").strip()
        if s:
            head = "\n".join(s.splitlines()[:20])
            pats = [
                re.compile(r"(?im)^\s*(?:exercise|problem|ex\.)?\s*(\d+)\.(\d+)\b"),
                re.compile(r"(?i)\b(?:exercise|problem|ex\.)\s*(\d+)\.(\d+)\b"),
            ]
            for p in pats:
                m = p.search(head)
                if m:
                    try:
                        return int(m.group(1)), int(m.group(2))
                    except Exception:
                        pass
        return int(fallback_chap), int(fallback_ex)

    def classify_problem_type(problem: str, proof: str) -> str:
        """
        Restrict output type to exactly one of:
        - 求值题
        - 证明题
        - 其他
        """
        p = (problem or "").strip()
        prf = (proof or "").strip()

        # Cues observed in optimization papers:
        # show/prove/verify are dominant proof-style tasks.
        proof_patterns = [
            r"\bshow\s+that\b",
            r"\bprove\b",
            r"\bverify\s+that\b",
            r"\bdemonstrate\s+that\b",
            r"\bestablish\s+that\b",
            r"\bif\s+and\s+only\s+if\b",
            r"证明",
            r"证(明)?",
            r"验证",
        ]
        for pat in proof_patterns:
            if re.search(pat, p, flags=re.IGNORECASE):
                return "证明题"

        # Evaluation/derivation/computation style cues.
        eval_patterns = [
            r"求[^\n]{0,8}值",
            r"计算",
            r"求解",
            r"\bevaluate\b",
            r"\bcompute\b",
            r"\bcalculate\b",
            r"\bfind\b",
            r"\bdetermine\b",
            r"\bgive\s+an?\s+explicit\s+solution\b",
            r"\bderive\b",
            r"\bfind\s+the\s+dual\b",
            r"\bminimi[sz]e\b",
            r"\bmaximi[sz]e\b",
            r"\bsolve\b",
        ]
        for pat in eval_patterns:
            if re.search(pat, p, flags=re.IGNORECASE):
                return "求值题"

        # Fallback: let LLM think when rule cues are unclear.
        if type_client is not None and type_model:
            llm_label = llm_classify_problem_type(
                type_client,
                type_model,
                problem=p,
                proof=prf,
                max_tokens=type_max_tokens,
                cache_dir=type_cache_dir,
                cache_enabled=type_cache_enabled,
            )
            llm_norm = _normalize_problem_type_label(llm_label)
            if llm_norm:
                return llm_norm

        # Final backstop.
        if prf:
            return "证明题"

        return "其他"

    def split_text_by_subparts(text: str) -> List[Tuple[Optional[str], str]]:
        """
        Split text by sub-question markers like:
          (a) ...
          (b) ...
        Returns [(part, body), ...], where part is 'a'/'b'/... or None.
        If no markers are found, returns one chunk with part=None.
        """
        s = (text or "").strip()
        if not s:
            return [(None, "")]

        # Prefer markers that start at line-begin to avoid matching normal parenthetical usage.
        pat = _marker_pattern()
        ms = list(pat.finditer(s))
        if not ms:
            return [(None, s)]

        out: List[Tuple[Optional[str], str]] = []
        for i, m in enumerate(ms):
            part = _normalize_part(m)
            start = m.start()
            end = ms[i + 1].start() if i + 1 < len(ms) else len(s)
            chunk = s[start:end].strip()
            if chunk:
                out.append((part, chunk))

        if not out:
            return [(None, s)]
        return out

    def split_problem_proof_subparts(problem: str, proof: str) -> List[Tuple[Optional[str], str, str]]:
        # Custom rule for solution manuals:
        # 1) Split by (a)/(b)/(c)... when present.
        # 2) Repeat common preface (text before first marker) for each split sub-question.
        p_text = (problem or "").strip()
        p_marker_re = _marker_pattern()
        p_markers = list(p_marker_re.finditer(p_text))
        p_prefix = ""
        if p_markers:
            p_prefix = (p_text[: p_markers[0].start()] or "").strip()

        p_parts = split_text_by_subparts(problem)
        # If problem has no subparts, do not split.
        if len(p_parts) <= 1 or p_parts[0][0] is None:
            return [(None, (problem or "").strip(), (proof or "").strip())]

        prf_parts_raw = split_text_by_subparts(proof)
        prf_text = (proof or "").strip()
        prf_marker_re = _marker_pattern()
        prf_markers = list(prf_marker_re.finditer(prf_text))
        prf_prefix = ""
        if prf_markers:
            prf_prefix = (prf_text[: prf_markers[0].start()] or "").strip()

        prf_map: Dict[str, str] = {}
        if prf_parts_raw and prf_parts_raw[0][0] is None:
            whole = (prf_parts_raw[0][1] or "").strip()
            for part, _ in p_parts:
                if part:
                    prf_map[part] = whole
        else:
            for part, body in prf_parts_raw:
                if part:
                    prf_map[part] = (body or "").strip()

        # Warn when proof has subparts but doesn't fully match problem subparts.
        if prf_markers:
            p_keys = [part for part, _ in p_parts if part]
            missing = [k for k in p_keys if k not in prf_map]
            if missing:
                print(f"[warn] proof subparts missing for: {', '.join(missing)}")

        out: List[Tuple[Optional[str], str, str]] = []
        for part, pbody in p_parts:
            pbody2 = (pbody or "").strip()
            # Repeat the common preface (global requirement) for every sub-question.
            if p_prefix:
                pbody2 = (p_prefix + "\n\n" + pbody2).strip()
            prf_body = prf_map.get((part or "").lower(), "").strip() if part else ""
            # Repeat common proof preface for every split proof sub-question.
            if prf_prefix and prf_body:
                prf_body = (prf_prefix + "\n\n" + prf_body).strip()
            out.append((part, pbody2, prf_body))
        return out

    fixed = (source_fixed or "").strip()

    out: List[Dict[str, Any]] = []
    out_index = 0
    for i, it in enumerate(items, start=1):
        problem = str(it.get("content") or "")
        proof = str(it.get("proof") or "")
        try:
            base_index = int(it.get("index", i))
        except Exception:
            base_index = i
        base_source_idx = ""
        ctx_chp = 0
        try:
            fallback_ex = int(base_index)
        except Exception:
            fallback_ex = i

        chp2, ex2 = extract_problem_number(problem, ctx_chp, fallback_ex)
        if chp2 > 0 and ex2 > 0:
            base_source_idx = f"Chp.{chp2} Ex.{ex2}"

        parts = split_problem_proof_subparts(problem, proof)
        for part, sub_problem, sub_proof in parts:
            out_index += 1
            source_idx = base_source_idx
            if part and source_idx:
                source_idx = f"{source_idx}-({part})"
            source_value = fixed if fixed else source_idx
            qtype = classify_problem_type(sub_problem, sub_proof)
            direct_answer = ""
            if qtype == "求值题" and answer_client is not None and answer_model:
                direct_answer = llm_extract_direct_answer(
                    answer_client,
                    answer_model,
                    problem=sub_problem,
                    proof=sub_proof,
                    max_tokens=answer_max_tokens,
                    cache_dir=answer_cache_dir,
                    cache_enabled=answer_cache_enabled,
                )

            out.append(
                {
                    "index": out_index,
                    "problem": sub_problem,
                    "proof": sub_proof,
                    "direct_answer": direct_answer,
                    "题目类型": qtype,
                    "预估难度": "",
                    "source": source_value,
                    "source_idx": source_idx,
                }
            )
    return out


# -----------------------------
# main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_tex", type=str, help="Input .tex")
    ap.add_argument("out_json", type=str, help="Output .json")
    ap.add_argument(
        "--source-fixed",
        type=str,
        default=None,
        help="Set a fixed value for output field 'source' for all items.",
    )
    ap.add_argument(
        "--implicit-deps",
        type=str,
        default=None,
        help="Implicit dependency recovery: off|rule|llm (default: settings TEXTOJSON_IMPLICIT_DEPS or 'rule')",
    )
    ap.add_argument(
        "--dep-review",
        type=str,
        default=None,
        help="LLM dependency review/repair: off|llm (default: settings TEXTOJSON_DEP_REVIEW or 'off')",
    )

    ap.add_argument(
        "--max-unit-chars",
        type=int,
        default=None,
        help="Max chars per LLM unit (default: settings TEXTOJSON_MAX_UNIT_CHARS or 6000)",
    )
    ap.add_argument(
        "--no-gen-proofs",
        action="store_true",
        help="Disable proof auto-generation for thm/lem/prop items with empty proof in source.",
    )
    ap.add_argument(
        "--refine",
        type=str,
        default=None,
        help="Refine statements/proofs into a more formal style: off|auto|all (default: settings TEXTOJSON_REFINE_MODE or 'off')",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable local on-disk cache for LLM responses (token saving across re-runs).",
    )
    args = ap.parse_args()

    cfg = load_config()
    settings = load_settings()
    api_key = require_str(cfg, "api_key")
    base_url = get_cfg(cfg, "base_url", "https://aihubmix.com/v1")
    # Optional: some providers require stream=True
    global STREAM_REQUIRED
    stream_mode = _coerce_stream_mode(get_cfg(cfg, "stream", None))
    if stream_mode is None:
        stream_mode = _coerce_stream_mode(get_cfg(cfg, "force_stream", None))
    STREAM_REQUIRED = stream_mode

    json_model = require_str(cfg, "model")
    max_tokens_json = int(get_setting(settings, "TEXTOJSON_MAX_TOKENS", 4096))

    refine_model = json_model
    max_tokens_refine = int(get_setting(settings, "TEXTOJSON_MAX_TOKENS_REFINE", 2048))

    refine_default = str(get_setting(settings, "TEXTOJSON_REFINE_MODE", "off")).strip().lower()
    refine_mode = (args.refine or refine_default).strip().lower()
    if refine_mode not in {"off", "auto", "all"}:
        refine_mode = "auto"

    cache_enabled = bool(get_setting(settings, "TEXTOJSON_CACHE_ENABLED", True)) and (
        not bool(getattr(args, "no_cache", False))
    )

    implicit_default = str(get_setting(settings, "TEXTOJSON_IMPLICIT_DEPS", "rule")).strip().lower()
    implicit_mode = (args.implicit_deps or implicit_default).strip().lower()
    if implicit_mode not in {"off", "rule", "llm"}:
        implicit_mode = "rule"

    dep_review_default = str(get_setting(settings, "TEXTOJSON_DEP_REVIEW", "off")).strip().lower()
    dep_review_mode = (args.dep_review or dep_review_default).strip().lower()
    if dep_review_mode not in {"off", "llm"}:
        dep_review_mode = "off"
    dep_patch_model = json_model
    max_tokens_dep_patch = int(get_setting(settings, "TEXTOJSON_MAX_TOKENS_DEP_PATCH", 1024))
    type_llm_fallback = bool(get_setting(settings, "TEXTOJSON_TYPE_LLM_FALLBACK", True))
    type_max_tokens = int(get_setting(settings, "TEXTOJSON_MAX_TOKENS_TYPE", 64))
    direct_answer_llm = bool(get_setting(settings, "TEXTOJSON_DIRECT_ANSWER_LLM", True))
    direct_answer_max_tokens = int(get_setting(settings, "TEXTOJSON_MAX_TOKENS_DIRECT_ANSWER", 128))

    max_unit_chars = int(get_setting(settings, "TEXTOJSON_MAX_UNIT_CHARS", 6000))
    if args.max_unit_chars is not None and args.max_unit_chars > 1000:
        max_unit_chars = int(args.max_unit_chars)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=int(get_setting(settings, "TEXTOJSON_TIMEOUT", 180)),
    )

    in_tex = Path(args.in_tex).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tex = in_tex.read_text(encoding="utf-8")

    items = convert_tex_to_items(
        tex,
        client=client,
        json_model=json_model,
        max_tokens_json=max_tokens_json,
        implicit_mode=implicit_mode,
        max_unit_chars=max_unit_chars,
        refine_mode=refine_mode,
        refine_model=refine_model,
        max_tokens_refine=max_tokens_refine,
        cache_enabled=cache_enabled,
        dep_review_mode=dep_review_mode,
        dep_patch_model=dep_patch_model,
        max_tokens_dep_patch=max_tokens_dep_patch,
    )

    source_fixed = args.source_fixed
    if SOURCE_NAME_ENTRY.strip():
        source_fixed = SOURCE_NAME_ENTRY.strip()
    elif source_fixed is None:
        source_fixed = get_setting(settings, "TEXTOJSON_SOURCE_FIXED", "")

    type_cache_dir = _default_cache_dir() if cache_enabled else None

    out_items = to_example_output_schema(
        items,
        source_fixed=source_fixed,
        type_client=(client if type_llm_fallback else None),
        type_model=(json_model if type_llm_fallback else None),
        type_max_tokens=type_max_tokens,
        type_cache_enabled=cache_enabled,
        type_cache_dir=type_cache_dir,
        answer_client=(client if direct_answer_llm else None),
        answer_model=(json_model if direct_answer_llm else None),
        answer_max_tokens=direct_answer_max_tokens,
        answer_cache_enabled=cache_enabled,
        answer_cache_dir=type_cache_dir,
    )
    out_json.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE: {out_json} (items={len(out_items)})")


if __name__ == "__main__":
    main()
