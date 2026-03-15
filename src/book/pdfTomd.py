#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Book-oriented PDF to Markdown OCR."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps
import pdf2image
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

STREAM_REQUIRED: Optional[bool] = None


OCR_MD_PROMPT = """
You are doing OCR for a math solutions manual page (Convex Optimization style).
Return Markdown only, with faithful transcription of visible page content.

Rules:
1) Do NOT add any explanation, comments, or instruction text.
2) Keep exercise headers/indices exactly when visible (e.g., "1.5 ...", "2.13 ...").
3) Keep sub-questions markers (a), (b), (c), ... when visible.
4) Keep "Solution." / "Proof." markers when visible; do not invent new solutions.
5) Preserve math faithfully using standard Markdown math:
   - inline: $...$
   - display: $$...$$
6) Do NOT hallucinate or expand repeated content. If unreadable, keep minimal placeholder [illegible].
7) Output only page content, no markdown code fences.
""".strip()

OCR_MD_PROMPT_FALLBACK = """
Conservative OCR mode.
Transcribe only clearly visible text and math from the image to Markdown.
Do not infer missing content, do not repeat lines, do not output any instructions.
Keep numbering and (a)(b)(c) markers when visible.
Output only content.
""".strip()


# -------- config helpers --------

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


def get_cfg(
    cfg: Dict[str, Any],
    key: str,
    default: Any,
    *,
    expected_type: Optional[type] = None,
    nonempty: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False,
) -> Any:
    """
    Typed config getter with optional range checks.
    Keeps config.json structure unchanged, but enforces types/ranges when requested.
    """
    v = cfg.get(key, default)
    if v is None and allow_none:
        return None

    if expected_type is not None and not isinstance(v, expected_type):
        raise TypeError(f"Invalid '{key}' in config.json: expected {expected_type.__name__}, got {type(v).__name__}")

    if isinstance(v, str) and nonempty and not v.strip():
        raise ValueError(f"Invalid '{key}' in config.json: must be non-empty string")

    if isinstance(v, (int, float)):
        if min_value is not None and float(v) < float(min_value):
            raise ValueError(f"Invalid '{key}' in config.json: {v} < min {min_value}")
        if max_value is not None and float(v) > float(max_value):
            raise ValueError(f"Invalid '{key}' in config.json: {v} > max {max_value}")

    return v


# -------- settings helpers (module-local JSON) --------

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


def get_setting(settings: Dict[str, Any], name: str, default: Any) -> Any:
    return settings.get(name, default)


# -------- PDF render (stream per page) --------

def get_pdf_page_count(pdf_path: Path) -> int:
    info = pdf2image.pdfinfo_from_path(str(pdf_path))
    pages = info.get("Pages")
    if not isinstance(pages, int) or pages <= 0:
        raise RuntimeError(f"Failed to read PDF page count: {pdf_path}")
    return pages


def render_single_page(pdf_path: Path, dpi: int, page_1based: int) -> Image.Image:
    images = pdf2image.convert_from_path(
        str(pdf_path),
        dpi=dpi,
        fmt="png",
        first_page=page_1based,
        last_page=page_1based,
        thread_count=1,
    )
    if not images:
        raise RuntimeError(f"Failed to render page {page_1based} at dpi={dpi}")
    im = images[0]
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


# -------- image utilities --------

def pad_image(img: Image.Image, pad_top: float = 0.0, pad_other: float = 0.0, *, max_px: int = 120) -> Image.Image:
    """
    Add white padding (helps OCR not miss edge-touching glyphs).
    Padding is now proportional to image size by default.

    pad_top / pad_other:
      - if 0 < value <= 1: treated as ratio of image height (top) / min(w,h) (other)
      - if value > 1: treated as pixels (compat)
    """
    w, h = img.size

    def _to_px(v: float, *, base: int) -> int:
        if v <= 0:
            return 0
        if 0 < v <= 1:
            return int(round(base * v))
        return int(round(v))

    top_px = _to_px(float(pad_top), base=h)
    other_base = min(w, h)
    other_px = _to_px(float(pad_other), base=other_base)

    if max_px > 0:
        top_px = min(top_px, int(max_px))
        other_px = min(other_px, int(max_px))

    if top_px <= 0 and other_px <= 0:
        return img

    border = (other_px, top_px, other_px, other_px)  # left, top, right, bottom
    return ImageOps.expand(img, border=border, fill="white")


def save_debug_images(
    debug_dir: Path,
    page_i: int,
    raw_img: Image.Image,
    ocr_img: Image.Image,
    tag: str,
    *,
    jpeg_quality: int = 90,
) -> None:
    """
    Save images for debugging (JPEG only):
      - raw render
      - final OCR input (after padding)
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    q = int(max(1, min(95, jpeg_quality)))
    try:
        raw_img.save(debug_dir / f"page_{page_i:04d}_{tag}_raw.jpg", "JPEG", quality=q, optimize=True, progressive=True)
        ocr_img.save(debug_dir / f"page_{page_i:04d}_{tag}_ocr.jpg", "JPEG", quality=q, optimize=True, progressive=True)
    except Exception:
        pass


# -------- OCR helpers --------

def _default_upload_max_side_for_dpi(dpi: int) -> int:
    """
    Heuristic: larger DPI -> allow larger upload max_side.
    Clamped to keep payload manageable.
    """
    # 350 dpi -> ~2100; 450 -> ~2700; 600 -> ~3600 (clamp)
    m = int(round(dpi * 6))
    return max(1800, min(3600, m))


def _downscale_if_needed(img: Image.Image, max_side: Optional[int]) -> Image.Image:
    """
    Downscale image so that max(width, height) <= max_side (keeps aspect ratio).
    Uses high-quality resampling. If max_side is None or <=0, no-op.
    """
    if not max_side or max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def pil_image_to_data_url(
    img: Image.Image,
    *,
    fmt: str = "JPEG",                 # "PNG" | "JPEG"
    jpeg_quality: int = 85,            # 1..95
    max_side: Optional[int] = 2000,    # int or None
    grayscale: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Encode image to data URL for API upload.
    Returns (data_url, info) where info includes actual upload size/bytes.
    """
    orig_w, orig_h = img.size

    img2 = _downscale_if_needed(img, max_side=max_side)
    up_w, up_h = img2.size

    if grayscale:
        img2 = img2.convert("L")
    else:
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")

    buf = io.BytesIO()
    fmt_u = (fmt or "JPEG").upper()

    if fmt_u == "PNG":
        img2.save(buf, format="PNG", optimize=True)
        mime = "image/png"
        actual_fmt = "PNG"
        q = None
    else:
        q = int(max(1, min(95, int(jpeg_quality))))
        img2.save(
            buf,
            format="JPEG",
            quality=q,
            optimize=True,
            progressive=True,
        )
        mime = "image/jpeg"
        actual_fmt = "JPEG"

    raw_bytes = buf.getvalue()
    b64 = base64.b64encode(raw_bytes).decode("ascii")

    info: Dict[str, Any] = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "upload_w": up_w,
        "upload_h": up_h,
        "upload_mode": "L" if grayscale else "RGB",
        "upload_fmt": actual_fmt,
        "jpeg_quality": q,
        "max_side": max_side,
        "grayscale": grayscale,
        "upload_bytes": len(raw_bytes),
        "upload_b64_chars": len(b64),
    }

    return f"data:{mime};base64,{b64}", info


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:md|markdown)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


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

_INSTRUCTION_LINE_RE = re.compile(
    r"^\s*(?:\d+\)\s*)?do\s+not\s+use\b",
    re.IGNORECASE,
)


def looks_like_layout(md: str) -> bool:
    s = (md or "").strip()
    if not s:
        return True
    head = s[:1000]
    return bool(re.search(
        r"\b(text|equation|interline_equation|sub_title|title|figure|table)\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]",
        head
    ))


def strip_layout_boxes(md: str) -> str:
    lines = (md or "").splitlines()
    out = []
    for ln in lines:
        ln2 = re.sub(r"^\s*\w+\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]\s*", "", ln)
        if ln2.strip():
            out.append(ln2.rstrip())
    return "\n".join(out).strip()


def strip_instruction_boilerplate(md: str, max_lines: int = 80) -> Tuple[str, int, int]:
    """
    Remove leading repeated instruction lines (prompt leakage) like:
      "Do NOT use `\\newpage` ..." or "8) Do NOT use ..."
    Returns (cleaned_md, removed_instruction_lines, total_instruction_lines).
    """
    lines = (md or "").splitlines()
    if not lines:
        return md, 0, 0

    total_instr = sum(1 for ln in lines if _INSTRUCTION_LINE_RE.match(ln))
    if total_instr == 0:
        return md, 0, 0

    removed_instr = 0
    i = 0
    removed = 0
    while i < len(lines) and removed < max_lines:
        ln = lines[i]
        if not ln.strip():
            i += 1
            removed += 1
            continue
        if _INSTRUCTION_LINE_RE.match(ln):
            removed_instr += 1
            i += 1
            removed += 1
            continue
        break

    if removed_instr == 0:
        return md, 0, total_instr

    cleaned = "\n".join(lines[i:]).strip()
    return cleaned, removed_instr, total_instr


def compile_boilerplate_patterns(patterns: Any) -> List[re.Pattern]:
    """
    patterns: list[str] regex strings.
    """
    out: List[re.Pattern] = []
    if not patterns:
        return out
    if not isinstance(patterns, list):
        return out
    for p in patterns:
        if not isinstance(p, str) or not p.strip():
            continue
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return out


def strip_boilerplate(md: str, patterns: List[re.Pattern]) -> Tuple[str, int]:
    """
    Remove leading boilerplate lines that match any configured pattern.
    Returns (cleaned_md, removed_count).
    """
    lines = (md or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    removed = 0
    # remove up to 6 leading boilerplate lines (more flexible than fixed 3)
    while lines and removed < 6:
        head = lines[0]
        if any(rx.match(head) for rx in patterns):
            lines.pop(0)
            removed += 1
            while lines and not lines[0].strip():
                lines.pop(0)
        else:
            break

    return "\n".join(lines).strip(), removed


def has_prompt_leakage(s: str) -> bool:
    text = (s or "").lower()
    # Common instruction leakage fragments from OCR prompts / conversion prompts.
    leak_markers = [
        "the output must be latex",
        "output markdown only",
        "output only the content",
        "strict block rules",
        "placeholder tokens",
        "do not move placeholders",
        "extract all text and math",
        "you are doing ocr",
    ]
    return any(m in text for m in leak_markers)


def has_runaway_number_list(s: str) -> bool:
    # Catch pathological "1. 2. 3. ... 500." loops.
    return bool(re.search(r"(?:\b\d+\.\s*){120,}", s or ""))


def has_heavy_line_repetition(s: str) -> bool:
    lines = [(ln or "").strip() for ln in (s or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) < 12:
        return False

    counts: Dict[str, int] = {}
    for ln in lines:
        key = re.sub(r"\s+", " ", ln).lower()
        if len(key) < 24:
            continue
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return False
    max_rep = max(counts.values())
    # "same long line repeated many times" is almost always OCR degeneration.
    if max_rep >= 6:
        return True
    # Or one line dominates too much.
    if max_rep / max(1, len(lines)) >= 0.25:
        return True
    return False


def _bad_reason_penalty(reason: Optional[str]) -> int:
    penalties = {
        "empty": 5000,
        "prompt_echo": 4500,
        "instruction_list": 4400,
        "runaway_number_list": 4200,
        "line_repetition": 3800,
        "sub-loop": 3000,
        "excessive_quad": 2000,
        "excessive_newlines": 1500,
        "char_repetition": 1500,
        "unbalanced_braces": 900,
    }
    return penalties.get(reason or "", 1000 if reason else 0)


def score_candidate(md: str, flags: Dict[str, Any]) -> int:
    """
    Higher is better.
    Prefer non-bad outputs, then prefer richer but not absurdly long text.
    """
    text = (md or "").strip()
    score = 0

    if flags.get("bad"):
        score -= _bad_reason_penalty(flags.get("bad_reason"))
    else:
        score += 2000

    n = len(text)
    if n < 20:
        score -= 1000
    elif n < 80:
        score -= 300
    elif n <= 14000:
        score += min(1800, n // 6)
    else:
        # Too long on a single page usually means degeneration.
        score -= min(2200, (n - 14000) // 10)

    if flags.get("layout"):
        score -= 80

    # Additional penalties for common corruption patterns.
    if has_prompt_leakage(text):
        score -= 3000
    if has_runaway_number_list(text):
        score -= 3000
    if has_heavy_line_repetition(text):
        score -= 2200

    return int(score)


def pick_better_candidate(
    a_md: str,
    a_flags: Dict[str, Any],
    b_md: str,
    b_flags: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Returns (best_md, best_flags, picked_b).
    """
    sa = score_candidate(a_md, a_flags)
    sb = score_candidate(b_md, b_flags)
    if sb > sa:
        return b_md, b_flags, True
    return a_md, a_flags, False


def postprocess_and_assess(md: str, boilerplate_patterns: List[re.Pattern]) -> Tuple[str, Dict[str, Any]]:
    """
    Merge: strip_code_fences + layout stripping + configurable boilerplate stripping + bad detection.
    Returns (clean_md, meta_flags).
    """
    meta: Dict[str, Any] = {
        "layout": False,
        "boilerplate_removed_lines": 0,
        "bad": False,
        "bad_reason": None,
    }

    s = strip_code_fences(md or "")

    if looks_like_layout(s):
        meta["layout"] = True
        s2 = strip_layout_boxes(s)
        if len(s2) > 50:
            s = s2

    s, removed = strip_boilerplate(s, boilerplate_patterns)
    meta["boilerplate_removed_lines"] = removed
    s, instr_removed, instr_total = strip_instruction_boilerplate(s)
    if instr_total:
        meta["instruction_lines_total"] = instr_total
        meta["instruction_lines_removed"] = instr_removed

    s_stripped = (s or "").strip()

    # --- 1. Basic Check: Empty ---
    if not s_stripped:
        meta["bad"] = True
        meta["bad_reason"] = "empty"
        return "", meta

    head = s_stripped[:800].lower()
    
    # --- 2. Basic Check: Loop/Hallucination ---
    if "sub-sub-sub" in head:
        meta["bad"] = True
        meta["bad_reason"] = "sub-loop"
        return s_stripped, meta

    # --- 2.5 Instruction-list prompt leakage ---
    lines = [ln for ln in s_stripped.splitlines() if ln.strip()]
    if lines:
        instr_cnt = sum(1 for ln in lines if _INSTRUCTION_LINE_RE.match(ln))
        if instr_cnt >= 8 and (instr_cnt / max(1, len(lines))) >= 0.3:
            meta["bad"] = True
            meta["bad_reason"] = "instruction_list"
            return s_stripped, meta

    # --- 3. Prompt leakage / echo ---
    echo_phrases = [
        "ocr to markdown",
        "output markdown only",
        "output only the content",
        "extract all text and math",
    ]
    if (len(s_stripped) < 240 and any(head.startswith(p) for p in echo_phrases)) or has_prompt_leakage(s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "prompt_echo"
        return s_stripped, meta

    # --- 4. Repetition / loop degeneration ---
    if has_runaway_number_list(s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "runaway_number_list"
        return s_stripped, meta

    if has_heavy_line_repetition(s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "line_repetition"
        return s_stripped, meta

    # --- 5. Math degradation checks ---
    if re.search(r"(\\quad\s*){6,}", s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "excessive_quad"
        return s_stripped, meta
        
    if re.search(r"(\\\\\s*){10,}", s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "excessive_newlines"
        return s_stripped, meta

    if re.search(r"([a-zA-Z0-9\uff01-\uff5e])\1{19,}", s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "char_repetition"
        return s_stripped, meta

    open_braces = s_stripped.count('{')
    close_braces = s_stripped.count('}')
    if abs(open_braces - close_braces) > 15:
        meta["bad"] = True
        meta["bad_reason"] = "unbalanced_braces"
        return s_stripped, meta

    return s_stripped, meta


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
)
def ocr_image_to_markdown(
    client: OpenAI,
    model: str,
    img: Image.Image,
    boilerplate_patterns: List[re.Pattern],
    *,
    max_tokens: Optional[int] = None,
    upload_fmt: str = "JPEG",
    upload_jpeg_quality: int = 85,
    upload_max_side: Optional[int] = 2000,
    upload_grayscale: bool = False,
    api_semaphore: Optional[threading.Semaphore] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (markdown, meta) where meta includes flags about fallback/layout/boilerplate and upload info.
    """
    url, upload_info = pil_image_to_data_url(
        img,
        fmt=upload_fmt,                      # FIX: no hardcode
        jpeg_quality=upload_jpeg_quality,    # FIX: no hardcode
        max_side=upload_max_side,            # FIX: no hardcode
        grayscale=upload_grayscale,          # FIX: no hardcode
    )

    def _call(prompt: str) -> str:
        # limit API concurrency if semaphore is provided
        if api_semaphore is not None:
            api_semaphore.acquire()
        try:
            return _chat_complete_text(
                client,
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
        finally:
            if api_semaphore is not None:
                api_semaphore.release()

    meta: Dict[str, Any] = {
        "used_fallback": False,
        "fallback_improved": False,
        "upload_fmt": upload_fmt,
        "upload_jpeg_quality": int(upload_jpeg_quality),
        "upload_max_side": upload_max_side,
        "upload_grayscale": bool(upload_grayscale),
        "upload_info": upload_info,
    }

    t_api0 = time.perf_counter()
    raw = _call(OCR_MD_PROMPT)
    t_api1 = time.perf_counter()
    meta["t_api_primary_s"] = t_api1 - t_api0

    cleaned, flags = postprocess_and_assess(raw, boilerplate_patterns)
    meta.update(flags)
    primary_md, primary_flags = cleaned, dict(flags)

    if primary_flags.get("bad"):
        meta["used_fallback"] = True
        t_f0 = time.perf_counter()
        raw2 = _call(OCR_MD_PROMPT_FALLBACK)
        t_f1 = time.perf_counter()
        meta["t_api_fallback_s"] = t_f1 - t_f0

        cleaned2, flags2 = postprocess_and_assess(raw2, boilerplate_patterns)
        best_md, best_flags, picked_fallback = pick_better_candidate(
            primary_md, primary_flags, cleaned2, flags2
        )
        cleaned = best_md
        meta.update(best_flags)
        meta["fallback_improved"] = bool(picked_fallback)
        meta["fallback_primary_score"] = score_candidate(primary_md, primary_flags)
        meta["fallback_second_score"] = score_candidate(cleaned2, flags2)

    return cleaned.strip(), meta


# -------- main --------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=str, help="Input PDF")
    ap.add_argument("out_md", type=str, help="Output Markdown file")

    ap.add_argument("--max-tokens", type=int, default=None, help="Override settings.OCR_MAX_TOKENS (default: use settings)")
    ap.add_argument("--workers", type=int, default=None, help="Override settings.OCR_WORKERS (default: use settings)")
    ap.add_argument("--debug", action="store_true", help="Override settings.OCR_DEBUG=True for this run")
    ap.add_argument("--no-debug", action="store_true", help="Override settings.OCR_DEBUG=False for this run")
    args = ap.parse_args()

    cfg = load_config()
    api_key = require_str(cfg, "api_key")
    base_url = get_cfg(cfg, "base_url", "https://aihubmix.com/v1", expected_type=str, nonempty=True)
    model = require_str(cfg, "model")
    # Optional: some providers require stream=True
    global STREAM_REQUIRED
    stream_mode = _coerce_stream_mode(get_cfg(cfg, "stream", None))
    if stream_mode is None:
        stream_mode = _coerce_stream_mode(get_cfg(cfg, "force_stream", None))
    STREAM_REQUIRED = stream_mode

    settings = load_settings()

    # dpi moved to settings.json
    dpi = int(get_setting(settings, "OCR_DPI", 350))
    if dpi < 72 or dpi > 1200:
        raise ValueError(f"Invalid OCR_DPI in settings.json: {dpi} (expected 72..1200)")


    pdf_path = Path(args.pdf).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # ---- settings ----
    workers = int(get_setting(settings, "OCR_WORKERS", 4))
    api_conc = int(get_setting(settings, "OCR_API_CONCURRENCY", max(1, min(4, workers))))
    timeout_s = int(get_setting(settings, "OCR_TIMEOUT", 120))
    max_tokens_setting = get_setting(settings, "OCR_MAX_TOKENS", None)

    debug_enabled = bool(get_setting(settings, "OCR_DEBUG", False))
    verbose_page_log = bool(get_setting(settings, "OCR_VERBOSE_PAGE_LOG", True))
    debug_jpeg_quality = int(get_setting(settings, "OCR_DEBUG_JPEG_QUALITY", 90))

    # padding (proportional)
    pad_top = float(get_setting(settings, "OCR_PAD_TOP", 0.015))
    pad_other = float(get_setting(settings, "OCR_PAD_OTHER", 0.004))
    pad_max_px = int(get_setting(settings, "OCR_PAD_MAX_PX", 80))

    # upload encoding controls
    upload_fmt = str(get_setting(settings, "OCR_UPLOAD_FMT", "JPEG")).upper()
    upload_jpeg_quality = int(get_setting(settings, "OCR_UPLOAD_JPEG_QUALITY", 85))
    upload_max_side_raw = get_setting(settings, "OCR_UPLOAD_MAX_SIDE", "auto")
    upload_grayscale = bool(get_setting(settings, "OCR_UPLOAD_GRAYSCALE", False))

    # boilerplate patterns (configurable)
    boilerplate_patterns = compile_boilerplate_patterns(
        get_setting(settings, "OCR_STRIP_PREFIX_PATTERNS", [])
    )

    # ---- CLI overrides ----
    if args.workers is not None:
        workers = int(args.workers)
    if workers <= 0:
        workers = 4

    if args.max_tokens is not None:
        ocr_max_tokens = int(args.max_tokens)
    else:
        ocr_max_tokens = max_tokens_setting
        if ocr_max_tokens is not None:
            ocr_max_tokens = int(ocr_max_tokens)

    if args.debug:
        debug_enabled = True
    if args.no_debug:
        debug_enabled = False

    api_conc = max(1, int(api_conc))
    if api_conc > workers:
        # allow, but usually you want api_conc <= workers
        pass

    # resolve upload_max_side (dpi-dynamic "auto")
    upload_max_side: Optional[int]
    if upload_max_side_raw is None:
        upload_max_side = None
    elif isinstance(upload_max_side_raw, str) and upload_max_side_raw.strip().lower() == "auto":
        upload_max_side = _default_upload_max_side_for_dpi(dpi)
    else:
        try:
            upload_max_side = int(upload_max_side_raw)  # may raise
        except Exception:
            upload_max_side = _default_upload_max_side_for_dpi(dpi)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    api_sem = threading.Semaphore(api_conc)

    # Debug dirs
    debug_dir = out_md.parent / f"{out_md.stem}_debug"
    pages_dir = debug_dir / "pages"
    per_page_md_dir = debug_dir / "page_md"

    # Page count
    t0 = time.perf_counter()
    n_pages = get_pdf_page_count(pdf_path)
    t1 = time.perf_counter()
    print(f"[init] pdf={pdf_path.name} pages={n_pages} dpi={dpi} (pdfinfo {t1 - t0:.3f}s)")
    print(
        f"[run] model={model} workers={workers} api_concurrency={api_conc} timeout={timeout_s}s "
        f"upload={upload_fmt} q={upload_jpeg_quality} max_side={upload_max_side} gray={upload_grayscale} "
        f"pad_top={pad_top} pad_other={pad_other} pad_max_px={pad_max_px} debug={debug_enabled}"
    )

    results: Dict[int, str] = {}
    quality: Dict[int, Dict[str, Any]] = {}

    def process_page(page_i: int) -> Tuple[int, str, Dict[str, Any]]:
        t_page0 = time.perf_counter()

        # 1) render single page (Standard Quality)
        t_r0 = time.perf_counter()
        raw_img = render_single_page(pdf_path, dpi=dpi, page_1based=page_i)
        t_r1 = time.perf_counter()

        # 2) padding
        ocr_img = pad_image(raw_img, pad_top=pad_top, pad_other=pad_other, max_px=pad_max_px)

        # 3) debug save (jpeg only)
        if debug_enabled:
            save_debug_images(pages_dir, page_i, raw_img, ocr_img, tag=f"dpi{dpi}", jpeg_quality=debug_jpeg_quality)

        # 4) OCR (First Attempt)
        t_o0 = time.perf_counter()
        md, meta = ocr_image_to_markdown(
            client,
            model,
            ocr_img,
            boilerplate_patterns,
            max_tokens=ocr_max_tokens,
            upload_fmt=upload_fmt,
            upload_jpeg_quality=upload_jpeg_quality,
            upload_max_side=upload_max_side,
            upload_grayscale=upload_grayscale,
            api_semaphore=api_sem,
        )
        t_o1 = time.perf_counter()

        if meta.get("bad"):
            high_dpi = max(dpi + 150, 450) 
            tqdm.write(f"[Retry] Page {page_i} bad quality ({meta.get('bad_reason')}). Retrying at DPI {high_dpi}...")

            try:
                raw_img_hq = render_single_page(pdf_path, dpi=high_dpi, page_1based=page_i)
                ocr_img_hq = pad_image(raw_img_hq, pad_top=pad_top, pad_other=pad_other, max_px=pad_max_px)

                if debug_enabled:
                    save_debug_images(pages_dir, page_i, raw_img_hq, ocr_img_hq, tag=f"dpi{high_dpi}_RETRY", jpeg_quality=debug_jpeg_quality)

                md_retry, meta_retry = ocr_image_to_markdown(
                    client,
                    model,
                    ocr_img_hq,
                    boilerplate_patterns,
                    max_tokens=ocr_max_tokens,
                    upload_fmt=upload_fmt,
                    upload_jpeg_quality=upload_jpeg_quality,
                    upload_max_side=_default_upload_max_side_for_dpi(high_dpi),
                    upload_grayscale=upload_grayscale,
                    api_semaphore=api_sem,
                )

                first_md = md
                first_meta = dict(meta)
                first_score = score_candidate(first_md, first_meta)
                retry_score = score_candidate(md_retry, meta_retry)

                best_md, best_flags, picked_retry = pick_better_candidate(
                    first_md,
                    first_meta,
                    md_retry,
                    meta_retry,
                )
                md = best_md
                if picked_retry:
                    meta_retry["is_retry"] = True
                    meta_retry["retry_reason"] = first_meta.get("bad_reason")
                    meta_retry["retry_improved"] = True
                    meta_retry["retry_primary_score"] = first_score
                    meta_retry["retry_second_score"] = retry_score
                    meta = meta_retry
                else:
                    meta["is_retry"] = True
                    meta["retry_reason"] = first_meta.get("bad_reason")
                    meta["retry_improved"] = False
                    meta["retry_primary_score"] = first_score
                    meta["retry_second_score"] = retry_score
                
            except Exception as e:
                tqdm.write(f"[Retry Failed] Page {page_i}: {e}")
        t_page1 = time.perf_counter()

        out_meta: Dict[str, Any] = {
            "page": page_i,
            "dpi": dpi,
            "t_render_s": t_r1 - t_r0,
            "t_ocr_total_s": t_o1 - t_o0,
            "t_page_total_s": t_page1 - t_page0,
            "md_len": len(md),
            "full_meta": meta,
            "upload": meta.get("upload_info", {}),
        }

        return page_i, md, out_meta

    # Run with proper tqdm (no broken redraw). Use tqdm.write for per-page logs.
    page_indices = list(range(1, n_pages + 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_page, i): i for i in page_indices}
        with tqdm(total=n_pages, desc="OCR pages", dynamic_ncols=True) as pbar:
            for fut in as_completed(futs):
                page_i, md, meta = fut.result()
                results[page_i] = md
                quality[page_i] = meta
                pbar.update(1)

                if verbose_page_log:
                    fm = meta.get("full_meta") or {}
                    bad = bool(fm.get("bad"))
                    layout = bool(fm.get("layout"))
                    used_fallback = bool(fm.get("used_fallback"))
                    up = meta.get("upload") or {}
                    up_bytes = int(up.get("upload_bytes") or 0)
                    up_kb = up_bytes / 1024.0
                    up_wh = f"{up.get('upload_w')}x{up.get('upload_h')}"
                    orig_wh = f"{up.get('orig_w')}x{up.get('orig_h')}"
                    up_fmt2 = up.get("upload_fmt")
                    up_q2 = up.get("jpeg_quality")
                    up_gray2 = up.get("grayscale")

                    # tqdm.write(
                    #     f"[OCR] page={page_i} bad={bad} layout={layout} fallback={used_fallback} len={len(md)} "
                    #     f"render={meta.get('t_render_s', 0):.3f}s ocr={meta.get('t_ocr_total_s', 0):.3f}s "
                    #     f"upload={up_fmt2} q={up_q2} gray={up_gray2} orig={orig_wh} up={up_wh} bytes={up_kb:.1f}KB"
                    # )

    # Assemble output in order
    chunks: List[str] = []
    for i in range(1, n_pages + 1):
        md = results.get(i, "")
        chunks.append(f"<!-- PAGE {i} -->\n{md}\n")
    out_md.write_text("\n".join(chunks), encoding="utf-8")

    if debug_enabled:
        debug_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "pdf": str(pdf_path),
            "out_md": str(out_md),
            "dpi": dpi,
            "workers": workers,
            "api_concurrency": api_conc,
            "timeout_s": timeout_s,
            "model": model,
            "pad_top": pad_top,
            "pad_other": pad_other,
            "pad_max_px": pad_max_px,
            "upload_fmt": upload_fmt,
            "upload_jpeg_quality": upload_jpeg_quality,
            "upload_max_side": upload_max_side,
            "upload_grayscale": upload_grayscale,
            "strip_prefix_patterns": [rx.pattern for rx in boilerplate_patterns],
            "pages": [quality[k] for k in sorted(quality.keys())],
        }
        (debug_dir / "quality_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"DONE: {out_md}")
    if debug_enabled:
        print(f"DEBUG: {debug_dir}")


if __name__ == "__main__":
    main()
