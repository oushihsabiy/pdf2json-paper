#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert OCR Markdown from pdfTomd.py into compile-ready LaTeX."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---- block-level detectors ----

RE_PAGE_MARK = re.compile(r"^\s*<!--\s*PAGE\s+\d+\s*-->\s*$", re.IGNORECASE)
RE_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RE_IMAGE = re.compile(r"^\s*!\[(.*?)\]\((.*?)\)\s*$")
RE_BULLET = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")
RE_NUMBERED = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")
RE_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


# ---- theorem/proof detectors ----

RE_THEOREM_HEADER = re.compile(
    r"^\s*(?:\*\*)?(Theorem|Lemma|Proposition|Definition|Corollary)"
    r"\s+(\d+(?:\.\d+)*)"
    r"(?:\s*\([^)]*\))?(?:\*\*)?\s*[:\.-]?\s*(.*)$",
    re.IGNORECASE,
)

RE_PROOF_HEADER = re.compile(
    r"^\s*(?:\*\*)?\*?(Proof(?:\s+of\s+[^*:.]+)?)\*?(?:\*\*)?\s*[:\.]?\s*(.*)$",
    re.IGNORECASE,
)

THEOREM_ENV_MAP = {
    "theorem": "theorem",
    "lemma": "lemma",
    "proposition": "proposition",
    "definition": "definition",
    "corollary": "corollary",
}


# ---- io ----


def read_markdown(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input Markdown file not found: {path}")
    if path.suffix.lower() != ".md":
        raise ValueError(f"Input file must be a .md file: {path}")
    return path.read_text(encoding="utf-8")


def write_tex(path: Path, latex_content: str) -> None:
    out_name = path.name.lower()
    if not (out_name.endswith(".tex") or out_name.endswith(".tex.tmp")):
        raise ValueError(f"Output file must be a .tex or .tex.tmp file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex_content, encoding="utf-8")


# ---- inline converters ----


def _split_inline_math(text: str) -> List[Tuple[str, bool]]:
    parts: List[Tuple[str, bool]] = []
    token_re = re.compile(r"(\$[^$\n]+\$)")
    last = 0
    for m in token_re.finditer(text):
        if m.start() > last:
            parts.append((text[last:m.start()], False))
        parts.append((m.group(1), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))
    return parts


def _escape_text(text: str) -> str:
    return re.sub(r"(?<!\\)([%&_#])", r"\\\1", text)


def _convert_links(text: str) -> str:
    return re.sub(r"\[(.+?)\]\((.+?)\)", r"\\href{\2}{\1}", text)


def _convert_emphasis(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", text)
    return text


def convert_inline_text(text: str) -> str:
    chunks = _split_inline_math(text)
    out: List[str] = []
    for piece, is_math in chunks:
        if is_math:
            out.append(piece)
            continue
        piece = _convert_links(piece)
        piece = _convert_emphasis(piece)
        piece = _escape_text(piece)
        out.append(piece)
    return "".join(out)


# ---- markdown parser ----


def parse_blocks(markdown: str) -> List[Dict[str, str]]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: List[Dict[str, str]] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if RE_PAGE_MARK.match(stripped):
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines: List[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            blocks.append({"type": "code", "content": "\n".join(code_lines)})
            continue

        if stripped.startswith("$$"):
            if stripped != "$$" and stripped.endswith("$$") and len(stripped) > 4:
                blocks.append({"type": "equation", "content": stripped[2:-2].strip()})
                i += 1
                continue

            eq_lines: List[str] = []
            i += 1
            while i < n and lines[i].strip() != "$$":
                eq_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            blocks.append({"type": "equation", "content": "\n".join(eq_lines).strip()})
            continue

        if RE_HEADING.match(stripped):
            blocks.append({"type": "heading", "content": stripped})
            i += 1
            continue

        if RE_IMAGE.match(stripped):
            blocks.append({"type": "image", "content": stripped})
            i += 1
            continue

        if i + 1 < n and "|" in line and RE_TABLE_SEP.match(lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            blocks.append({"type": "table", "content": "\n".join(table_lines)})
            continue

        if line.lstrip().startswith(">"):
            qlines: List[str] = []
            while i < n and lines[i].lstrip().startswith(">"):
                qlines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            blocks.append({"type": "quote", "content": "\n".join(qlines).strip()})
            continue

        if RE_BULLET.match(line):
            items: List[str] = []
            while i < n and RE_BULLET.match(lines[i]):
                m = RE_BULLET.match(lines[i])
                items.append(m.group(1) if m else lines[i].strip())
                i += 1
            blocks.append({"type": "itemize", "content": "\n".join(items)})
            continue

        if RE_NUMBERED.match(line):
            items = []
            while i < n and RE_NUMBERED.match(lines[i]):
                m = RE_NUMBERED.match(lines[i])
                items.append(m.group(1) if m else lines[i].strip())
                i += 1
            blocks.append({"type": "enumerate", "content": "\n".join(items)})
            continue

        para_lines = [line]
        i += 1
        while i < n:
            cur = lines[i]
            cur_s = cur.strip()
            if not cur_s:
                break
            if RE_PAGE_MARK.match(cur_s):
                break
            if cur_s.startswith("```") or cur_s.startswith("$$"):
                break
            if RE_HEADING.match(cur_s) or RE_IMAGE.match(cur_s):
                break
            if cur.lstrip().startswith(">"):
                break
            if RE_BULLET.match(cur) or RE_NUMBERED.match(cur):
                break
            if i + 1 < n and "|" in cur and RE_TABLE_SEP.match(lines[i + 1]):
                break
            para_lines.append(cur)
            i += 1

        blocks.append({"type": "paragraph", "content": "\n".join(para_lines).strip()})

    return blocks


# ---- block converters ----


def convert_heading(line: str) -> str:
    m = RE_HEADING.match(line)
    if not m:
        return convert_inline_text(line)

    level = len(m.group(1))
    title = convert_inline_text(m.group(2).strip())
    if level == 1:
        return f"\\section{{{title}}}"
    if level == 2:
        return f"\\subsection{{{title}}}"
    return f"\\subsubsection{{{title}}}"


def convert_equation(content: str) -> str:
    body = content.strip()
    if "\\\\" in body:
        return "\\begin{align}\n" + body + "\n\\end{align}"
    return "\\begin{equation}\n" + body + "\n\\end{equation}"


def convert_list(block_type: str, content: str) -> str:
    env = "itemize" if block_type == "itemize" else "enumerate"
    items = [ln for ln in content.split("\n") if ln.strip()]
    body = "\n".join(f"\\item {convert_inline_text(it.strip())}" for it in items)
    return f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"


def convert_image(line: str) -> str:
    m = RE_IMAGE.match(line)
    if not m:
        return convert_inline_text(line)

    caption = convert_inline_text(m.group(1).strip())
    path = m.group(2).strip()
    return (
        "\\begin{figure}[h]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.8\\textwidth]{{{path}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}"
    )


def _split_table_row(row: str) -> List[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def convert_table(content: str) -> str:
    rows = [ln for ln in content.split("\n") if ln.strip()]
    if len(rows) < 2:
        return convert_inline_text(content)

    header = _split_table_row(rows[0])
    body_rows = [_split_table_row(r) for r in rows[2:]]
    col_count = len(header)
    if col_count == 0:
        return convert_inline_text(content)

    header_line = " & ".join(convert_inline_text(c) for c in header) + r" \\"
    body_lines: List[str] = []
    for row in body_rows:
        normalized = (row + [""] * col_count)[:col_count]
        body_lines.append(" & ".join(convert_inline_text(c) for c in normalized) + r" \\")

    lines = [
        f"\\begin{{tabular}}{{{'l' * col_count}}}",
        "\\toprule",
        header_line,
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def convert_quote(content: str) -> str:
    text = "\n".join(convert_inline_text(ln) for ln in content.split("\n"))
    return "\\begin{quote}\n" + text + "\n\\end{quote}"


# ---- theorem/proof conversion ----


def _strip_single_asterisk_pair(text: str) -> str:
    text = text.strip()
    return re.sub(r"^\*(.+)\*$", r"\1", text)


def _parse_theorem_header(line: str) -> Optional[Tuple[str, str]]:
    m = RE_THEOREM_HEADER.match(line.strip())
    if not m:
        return None

    kind = m.group(1).lower()
    env = THEOREM_ENV_MAP.get(kind, "theorem")
    first_tail = _strip_single_asterisk_pair((m.group(3) or "").strip())
    return env, first_tail


def _parse_proof_header(line: str) -> Optional[str]:
    m = RE_PROOF_HEADER.match(line.strip())
    if not m:
        return None
    return _strip_single_asterisk_pair((m.group(2) or "").strip())


def convert_math_environments(paragraph: str) -> str:
    lines = [ln.rstrip() for ln in paragraph.split("\n") if ln.strip()]
    if not lines:
        return ""

    theorem_header = _parse_theorem_header(lines[0])
    if theorem_header is not None:
        env, first_rest = theorem_header

        theorem_lines: List[str] = [first_rest] if first_rest else []
        proof_lines: List[str] = []
        in_proof = False

        for ln in lines[1:]:
            if not in_proof:
                proof_start = _parse_proof_header(ln)
                if proof_start is not None:
                    in_proof = True
                    if proof_start:
                        proof_lines.append(proof_start)
                    continue
                theorem_lines.append(ln)
            else:
                proof_lines.append(ln)

        theorem_body = "\n".join(convert_inline_text(ln) for ln in theorem_lines).strip()
        theorem_block = f"\\begin{{{env}}}\n{theorem_body}\n\\end{{{env}}}"

        if proof_lines:
            proof_body = "\n".join(convert_inline_text(ln) for ln in proof_lines).strip()
            proof_block = f"\\begin{{proof}}\n{proof_body}\n\\end{{proof}}"
            return theorem_block + "\n\n" + proof_block

        return theorem_block

    proof_header = _parse_proof_header(lines[0])
    if proof_header is not None:
        proof_content_lines: List[str] = [proof_header] if proof_header else []
        proof_content_lines.extend(lines[1:])
        proof_body = "\n".join(convert_inline_text(ln) for ln in proof_content_lines).strip()
        return f"\\begin{{proof}}\n{proof_body}\n\\end{{proof}}"

    return ""


def convert_paragraph(content: str) -> str:
    env_block = convert_math_environments(content)
    if env_block:
        return env_block
    return "\n".join(convert_inline_text(ln) for ln in content.split("\n"))


# ---- document assembly ----


def assemble_latex_document(body: str) -> str:
    preamble = (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\usepackage{amsthm}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{booktabs}\n\n"
        "\\newtheorem{theorem}{Theorem}\n"
        "\\newtheorem{lemma}{Lemma}\n"
        "\\newtheorem{proposition}{Proposition}\n"
        "\\newtheorem{corollary}{Corollary}\n"
        "\\theoremstyle{definition}\n"
        "\\newtheorem{definition}{Definition}\n\n"
        "\\begin{document}\n\n"
    )
    ending = "\n\\end{document}\n"
    return preamble + body.strip() + ending


def convert_markdown_to_latex(md_text: str) -> str:
    blocks = parse_blocks(md_text)
    out_blocks: List[str] = []

    for blk in blocks:
        kind = blk.get("type", "")
        content = blk.get("content", "")

        if kind == "heading":
            out_blocks.append(convert_heading(content))
        elif kind == "equation":
            out_blocks.append(convert_equation(content))
        elif kind in {"itemize", "enumerate"}:
            out_blocks.append(convert_list(kind, content))
        elif kind == "image":
            out_blocks.append(convert_image(content))
        elif kind == "table":
            out_blocks.append(convert_table(content))
        elif kind == "quote":
            out_blocks.append(convert_quote(content))
        elif kind == "code":
            out_blocks.append("\\begin{verbatim}\n" + content + "\n\\end{verbatim}")
        elif kind == "paragraph":
            out_blocks.append(convert_paragraph(content))
        else:
            out_blocks.append(convert_inline_text(content))

    body = "\n\n".join(x for x in out_blocks if x.strip())
    return assemble_latex_document(body)


# ---- cli ----


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Markdown from pdfTomd.py to compile-ready LaTeX for math/optimization papers."
    )
    parser.add_argument("input_md", help="Path to input Markdown file")
    parser.add_argument("output_tex", help="Path to output TeX file")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="(ignored) Worker count placeholder for pipeline compatibility",
    )
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()

    in_path = Path(args.input_md).expanduser().resolve()
    out_path = Path(args.output_tex).expanduser().resolve()

    try:
        md_text = read_markdown(in_path)
        latex_text = convert_markdown_to_latex(md_text)
        write_tex(out_path, latex_text)
        print(f"DONE: {out_path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
