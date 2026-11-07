"""Text extraction and cleaning for .pdf, .docx, .md and .txt files.

Produces a Document: an ordered list of Blocks (heading / para / list) with the
academic apparatus removed (references, captions, equations, affiliations,
acknowledgments, headers/footers, footnote markers) and every removal logged.
Paragraphs are tagged with a coarse rhetorical role.

CLI:  python extract.py PATH [PATH ...]   (files or directories; directories are
      searched recursively)  -> JSON summary on stdout, log on stderr.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log  # noqa: E402

SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}


@dataclass
class Block:
    text: str
    kind: str = "para"          # heading | para | list | protected (verbatim, never rewritten)
    level: int = 0              # heading level
    role: str = "other"         # intro | methods | results | discussion | other
    heading: str = ""           # nearest preceding heading
    src_index: int = -1         # paragraph index in the source .docx (for in-place output)


@dataclass
class Document:
    path: str
    blocks: list[Block] = field(default_factory=list)
    stripped: Counter = field(default_factory=Counter)   # what was removed, by category
    stripped_words: Counter = field(default_factory=Counter)
    citations: Counter = field(default_factory=Counter)  # numeric / author_year

    @property
    def paragraphs(self) -> list[Block]:
        return [b for b in self.blocks if b.kind in ("para", "list")]

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.paragraphs)

    @property
    def n_words(self) -> int:
        return sum(len(b.text.split()) for b in self.paragraphs)

    def summary(self) -> dict:
        return {
            "path": self.path,
            "words": self.n_words,
            "paragraphs": len(self.paragraphs),
            "headings": sum(b.kind == "heading" for b in self.blocks),
            "roles": dict(Counter(b.role for b in self.paragraphs)),
            "stripped": dict(self.stripped),
            "stripped_words": dict(self.stripped_words),
            "citations": dict(self.citations),
        }


# ====================================================================== files

def collect_files(paths) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED and not any(
                        part.startswith(".") for part in f.relative_to(p).parts):
                    out.append(f)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED:
                log(f"[extract] skipping unsupported file type: {p}")
            else:
                out.append(p)
        else:
            log(f"[extract] not found: {p}")
    # de-dup, keep order
    seen, uniq = set(), []
    for f in out:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(f)
    return uniq


# ====================================================================== readers

def dehyphenate(text: str) -> str:
    # "exam-\nple" -> "example" (only when the join gives a lowercase continuation)
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)
    text = text.replace("­", "")  # soft hyphen
    return text


def join_lines(text: str) -> str:
    text = dehyphenate(text)
    return re.sub(r"\s*\n\s*", " ", text).strip()


def read_pdf(path: Path, doc: Document) -> list[Block]:
    import pymupdf as fitz

    pdf = fitz.open(path)
    pages = []
    for page in pdf:
        h = page.rect.height
        items = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            lines, sizes, bold = [], [], []
            for ln in b["lines"]:
                t = "".join(s["text"] for s in ln["spans"])
                if t.strip():
                    lines.append(t)
                for s in ln["spans"]:
                    if s["text"].strip():
                        sizes.append(s["size"])
                        bold.append(bool(s["flags"] & 16))
            if not lines:
                continue
            y0, y1 = b["bbox"][1], b["bbox"][3]
            items.append({"lines": lines, "size": statistics.median(sizes),
                          "bold": sum(bold) > len(bold) / 2, "top": y0 / h, "bottom": y1 / h})
        pages.append(items)

    # headers/footers: short blocks near the page edges whose digit-normalised text
    # repeats on many pages, plus bare page numbers.
    def norm(s):
        return re.sub(r"\d+", "#", " ".join(s.split()).lower())

    edge_counts = Counter()
    for items in pages:
        for it in items:
            if it["top"] < 0.08 or it["bottom"] > 0.92:
                edge_counts[norm(" ".join(it["lines"]))] += 1
    repeat_min = max(2, int(0.3 * len(pages)))
    all_sizes = [it["size"] for items in pages for it in items]
    body_size = statistics.median(all_sizes) if all_sizes else 10

    blocks: list[Block] = []
    for items in pages:
        for it in items:
            raw = "\n".join(it["lines"])
            flat = " ".join(raw.split())
            at_edge = it["top"] < 0.08 or it["bottom"] > 0.92
            if at_edge and (re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", flat, re.I)
                            or edge_counts[norm(flat)] >= repeat_min) and len(flat.split()) < 20:
                doc.stripped["header_footer"] += 1
                doc.stripped_words["header_footer"] += len(flat.split())
                continue
            n_words = len(flat.split())
            is_heading = (n_words <= 12 and not re.search(r"[.:;,]$", flat)
                          and (it["size"] > body_size * 1.12 or it["bold"]))
            if is_heading:
                blocks.append(Block(flat, "heading", 1 if it["size"] > body_size * 1.3 else 2))
            else:
                blocks.append(Block(join_lines(raw), "para"))

    # merge paragraphs split across page/column breaks
    merged: list[Block] = []
    for b in blocks:
        if (merged and b.kind == "para" and merged[-1].kind == "para"
                and not re.search(r"[.!?:\"”)]$", merged[-1].text)
                and re.match(r"[a-z(]", b.text)):
            merged[-1].text = merged[-1].text + " " + b.text
        else:
            merged.append(b)
    return merged


def read_docx(path: Path, doc: Document) -> list[Block]:
    import docx

    d = docx.Document(str(path))
    blocks = []
    for pi, p in enumerate(d.paragraphs):
        t = " ".join(p.text.split())
        if not t:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading") or style == "title":
            m = re.search(r"\d+", style)
            blocks.append(Block(t, "heading", int(m.group()) if m else 1, src_index=pi))
        elif style == "caption":
            doc.stripped["caption"] += 1
            doc.stripped_words["caption"] += len(t.split())
            blocks.append(Block(t, "protected", role="protected", src_index=pi))
        elif "list" in style or p._p.pPr is not None and p._p.pPr.numPr is not None:
            blocks.append(Block(t, "list", src_index=pi))
        else:
            blocks.append(Block(t, "para", src_index=pi))
    if d.tables:
        doc.stripped["table"] += len(d.tables)
    return blocks


MD_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def clean_inline_md(t: str) -> str:
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)              # images
    t = re.sub(r"\[\^[^\]]+\]", "", t)                       # footnote refs
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)          # links
    t = re.sub(r"<[^>]+>", "", t)                            # html tags
    t = re.sub(r"\$[^$\n]+\$", "X", t)                       # inline math
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    return t


def read_markdown(text: str, doc: Document, keep: bool = False) -> list[Block]:
    """keep=True (rewrite mode): front matter, code, display math and tables become
    verbatim 'protected' blocks and inline markdown is left untouched."""
    protected: list[str] = []

    def stash(m):
        protected.append(m.group(0).strip("\n"))
        return f"\n\n\x00P{len(protected) - 1}\x00\n\n"

    fm = re.match(r"\A---\n.*?\n---\n", text, flags=re.S)
    if fm:
        text = (stash(fm) if keep else "") + text[fm.end():]
    n_code = len(re.findall(r"^```", text, flags=re.M)) // 2
    text = re.sub(r"^```.*?^```[^\n]*$", stash if keep else "", text, flags=re.S | re.M)
    n_math = len(re.findall(r"\$\$", text)) // 2
    text = re.sub(r"\$\$.*?\$\$", stash if keep else "", text, flags=re.S)
    if n_code:
        doc.stripped["code_block"] += n_code
    if n_math:
        doc.stripped["equation"] += n_math
    inline = (lambda t: t) if keep else clean_inline_md
    blocks = []
    for chunk in re.split(r"\n\s*\n", text):
        m = re.fullmatch(r"\s*\x00P(\d+)\x00\s*", chunk)
        if m:
            blocks.append(Block(protected[int(m.group(1))], "protected", role="protected"))
            continue
        lines = [ln for ln in chunk.strip("\n").split("\n") if ln.strip()]
        if not lines:
            continue
        if all(ln.strip().startswith("|") for ln in lines):
            doc.stripped["table"] += 1
            if keep:
                blocks.append(Block("\n".join(lines), "protected", role="protected"))
            continue
        if re.match(r"^#{1,6}\s", lines[0]):
            level = len(lines[0]) - len(lines[0].lstrip("#"))
            blocks.append(Block(clean_inline_md(lines[0].lstrip("#").strip()), "heading", level))
            lines = lines[1:]
            if not lines:
                continue
        if MD_LIST_RE.match(lines[0]):
            items, cur = [], []
            for ln in lines:
                if MD_LIST_RE.match(ln):
                    if cur:
                        items.append(" ".join(cur))
                    cur = [MD_LIST_RE.sub("", ln).strip()]
                else:
                    cur.append(ln.strip())
            if cur:
                items.append(" ".join(cur))
            blocks.extend(Block(inline(i), "list") for i in items)
            continue
        body = join_lines("\n".join(ln.lstrip("> ") for ln in lines))
        blocks.append(Block(inline(body), "para"))
    return blocks


def strip_gutenberg(text: str) -> str:
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if m:
        text = text[:m.start()]
    return text


def looks_like_heading(t: str) -> bool:
    words = t.split()
    if not words or len(words) > 12 or re.search(r"[.,;:!?\"”]$", t) and not re.fullmatch(r"[IVXLC\d]+\.", t):
        return False
    if re.match(r"^(\d+(\.\d+)*\.?|[IVXLC]+\.)\s*", t):
        return True
    if t.isupper() and any(c.isalpha() for c in t):
        return True
    caps = sum(w[0].isupper() for w in words if w[0].isalpha())
    alpha = sum(1 for w in words if w[0].isalpha())
    return alpha > 0 and caps / alpha >= 0.6 and len(words) <= 8


def read_text(text: str, doc: Document) -> list[Block]:
    text = strip_gutenberg(text)
    blocks = []
    for chunk in re.split(r"\n\s*\n", text):
        flat = " ".join(chunk.split())
        if not flat:
            continue
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        if len(lines) <= 2 and looks_like_heading(flat):
            blocks.append(Block(flat, "heading", 1))
        elif MD_LIST_RE.match(lines[0]) and all(MD_LIST_RE.match(ln) or ln.startswith(" ") for ln in lines):
            for ln in lines:
                if MD_LIST_RE.match(ln):
                    blocks.append(Block(MD_LIST_RE.sub("", ln).strip(), "list"))
                elif blocks:
                    blocks[-1].text += " " + ln.strip()
        else:
            blocks.append(Block(re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", join_lines(chunk)), "para"))
    return blocks


# ====================================================================== cleaning

REF_HEAD = re.compile(r"^(\d+(\.\d+)*\.?\s*|[IVX]+\.\s*)?(references|bibliography|works cited|"
                      r"literature cited|reference list|sources cited|notes and references)\s*$", re.I)
ACK_HEAD = re.compile(r"^(\d+\.?\s*)?(acknowledge?ments?|funding|author contributions|"
                      r"conflicts? of interest|declaration of competing interests?|data availability)\s*$", re.I)
ABSTRACT_HEAD = re.compile(r"^(abstract|summary)\b", re.I)
CAPTION_RE = re.compile(r"^(figure|fig\.|table|exhibit|chart|plate|scheme)\s*[A-Z]?\d+[a-z]?\s*[.:|—-]", re.I)
AFFIL_RE = re.compile(r"(@|\buniversity\b|\bdepartment\b|\binstitute\b|\bcollege\b|"
                      r"\bschool of\b|\blaborator(y|ies)\b|\bcorresponding author\b|orcid)", re.I)
MATH_CHARS = set("=+−×÷∑∫∂√≤≥≈∞αβγδθλμσπ^_{}\\|<>")
SUPERSCRIPTS = "¹²³⁰⁴⁵⁶⁷⁸⁹"

NUM_CIT = re.compile(r"\[(\d+([–,\-\s]+\d+)*)\]")
AY_CIT = re.compile(r"\(([A-Z][A-Za-zÀ-ſ'\-]+( et al\.?| and [A-Z][A-Za-z'\-]+| & [A-Z][A-Za-z'\-]+)?,? "
                    r"\(?(1[6-9]|20)\d\d[a-z]?(, ?(p|pp)\. ?[\d–-]+)?;?\s*)+\)")
AY_NARR = re.compile(r"\b[A-Z][A-Za-z'\-]+( et al\.?| and [A-Z][A-Za-z'\-]+)? \((1[6-9]|20)\d\d[a-z]?\)")


def is_equation(t: str) -> bool:
    toks = t.split()
    if not toks:
        return False
    alpha_words = sum(1 for w in toks if re.fullmatch(r"[A-Za-z][a-z]+[.,;:]?", w))
    math = sum(c in MATH_CHARS for c in t)
    if re.search(r"\\(frac|sum|int|begin|alpha|beta|mathbf|left|right)", t):
        return True
    return (math >= 3 and alpha_words / len(toks) < 0.5) or (len(toks) <= 25 and math / max(1, len(t)) > 0.08)


def is_table_row(t: str) -> bool:
    toks = t.split()
    if len(toks) < 4:
        return False
    numeric = sum(bool(re.fullmatch(r"[\d.,%−\-±()]+", w)) for w in toks)
    return numeric / len(toks) > 0.5


def clean_document(doc: Document, blocks: list[Block], keep: bool = False) -> list[Block]:
    """Remove academic apparatus. With keep=True (rewrite mode) nothing is removed:
    the apparatus becomes 'protected' blocks that are passed through verbatim, and
    citations / footnote markers stay in the text."""
    def drop(cat, b) -> bool:
        """Record the removal; return True if the block should still be kept."""
        doc.stripped[cat] += 1
        doc.stripped_words[cat] += len(b.text.split())
        if keep:
            if b.kind != "heading":     # headings keep their level for reassembly
                b.kind = "protected"
            b.role = "protected"
            return True
        return False

    # references -> drop everything after the heading (appendices are usually
    # tables/proofs; any prose there is a small loss)
    for i, b in enumerate(blocks):
        if b.kind != "protected" and (b.kind == "heading" or len(b.text.split()) <= 4) \
                and REF_HEAD.match(b.text.strip()):
            for x in blocks[i:]:
                drop("references", x)
            if not keep:
                blocks = blocks[:i]
            break

    # author / affiliation block before the abstract or the first section
    first_n = min(len(blocks), 25)
    # the affiliation heuristic only applies to the front matter: blocks before the
    # first body paragraph of real length
    front_end = next((i for i, b in enumerate(blocks) if b.kind == "para" and len(b.text.split()) >= 60), 0)
    abs_idx = next((i for i, b in enumerate(blocks[:first_n]) if ABSTRACT_HEAD.match(b.text)), None)
    out = []
    for i, b in enumerate(blocks):
        if b.kind in ("para", "list") and abs_idx is not None and i < abs_idx:
            if drop("author_affiliation", b):
                out.append(b)
            continue
        if i < min(first_n, front_end) and b.kind == "para" and len(b.text.split()) < 40 and AFFIL_RE.search(b.text):
            if drop("author_affiliation", b):
                out.append(b)
            continue
        out.append(b)
    blocks = out

    # acknowledgments & similar back-matter sections
    out, skipping = [], False
    for b in blocks:
        if b.kind == "protected":
            out.append(b)
            continue
        if b.kind == "heading":
            skipping = bool(ACK_HEAD.match(b.text.strip()))
            if skipping:
                if drop("acknowledgments", b):
                    out.append(b)
                continue
        elif b.kind == "para" and re.match(r"^(acknowledge?ments?)[.:]", b.text, re.I):
            if drop("acknowledgments", b):
                out.append(b)
            continue
        if skipping:
            if drop("acknowledgments", b):
                out.append(b)
            continue
        out.append(b)
    blocks = out

    out = []
    for b in blocks:
        if b.kind in ("heading", "protected"):
            out.append(b)
            continue
        t = b.text
        cat = ("caption" if CAPTION_RE.match(t) else "equation" if is_equation(t) else
               "table" if is_table_row(t) else "page_number" if re.fullmatch(r"[\d\s\-\u2013]+", t) else None)
        if cat:
            if drop(cat, b):
                out.append(b)
            continue
        # citations are always counted
        doc.citations["numeric"] += len(NUM_CIT.findall(t))
        doc.citations["author_year"] += len(AY_CIT.findall(t)) + len(AY_NARR.findall(t))
        if keep:
            out.append(b)
            continue
        # footnote markers
        n_sup = sum(t.count(c) for c in SUPERSCRIPTS)
        t = re.sub(f"[{SUPERSCRIPTS}]+", "", t)
        t, n_fn = re.subn(r"(?<=[a-z][.,;:)\u201d\"])\d{1,2}(?=\s+[A-Z]|$)", "", t)
        if n_sup or n_fn:
            doc.stripped["footnote_marker"] += n_sup + n_fn
        # citations: remove so they don't distort sentence statistics
        t = NUM_CIT.sub("", t)
        t = AY_CIT.sub("", t)
        t = re.sub(r"\s+([.,;:])", r"\1", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        if t:
            b.text = t
            out.append(b)
    return out


# ====================================================================== roles

ROLE_HEADS = [
    ("intro", r"intro|background|motivation|overview|abstract|summary|preface|prologue|context|related work|literature review"),
    ("methods", r"method|materials|data\b|dataset|experimental (setup|design)|approach|design|procedure|"
                r"implementation|participants|setting|sample|measures|model\b|framework|study area"),
    ("results", r"result|finding|experiment|evaluation|analysis|performance|empirical"),
    ("discussion", r"discussion|conclusion|concluding|implication|limitation|future work|outlook|"
                   r"recommendation|epilogue|closing|final thoughts"),
]
CUE_RESULTS = re.compile(r"\b(we (found|observed)|results (show|indicate|suggest)|"
                         r"(statistically )?significant(ly)? (difference|effect|increase|decrease|"
                         r"correlation|association|interaction|predictor)s?|statistically significant|"
                         r"table \d|figure \d|fig\. \d|p ?[<=] ?0?\.\d|\bci\b|odds ratio|r\^?2 ?=)", re.I)
CUE_METHODS = re.compile(r"\b(we (used|collected|measured|recruited|trained|computed|estimated)|"
                         r"were (collected|measured|recruited|computed|estimated|sampled)|"
                         r"participants|dataset|procedure)\b", re.I)


def role_from_heading(h: str) -> str | None:
    h = re.sub(r"^[\d.\sIVX]+", "", h.lower())
    for role, pat in ROLE_HEADS:
        if re.search(pat, h):
            return role
    return None


def cue_role(text: str) -> str | None:
    if CUE_RESULTS.search(text):
        return "results"
    if CUE_METHODS.search(text):
        return "methods"
    return None


def tag_roles(blocks: list[Block]) -> None:
    current_role, current_head = None, ""
    paras = [b for b in blocks if b.kind in ("para", "list")]
    n = len(paras)
    k = 0
    for b in blocks:
        if b.kind == "heading":
            r = role_from_heading(b.text)
            current_head = b.text
            # a heading we cannot classify keeps the role of its parent section only
            # when it is a sub-heading
            if r is not None or b.level <= 1:
                current_role = r
            continue
        if b.kind == "protected":
            continue
        b.heading = current_head
        if current_role:
            b.role = current_role
        else:
            pos = k / max(1, n - 1) if n > 1 else 0.0
            if CUE_RESULTS.search(b.text):
                b.role = "results"
            elif CUE_METHODS.search(b.text):
                b.role = "methods"
            elif pos <= 0.12:
                b.role = "intro"
            elif pos >= 0.88:
                b.role = "discussion"
            else:
                b.role = "other"
        k += 1


# ====================================================================== main API

def extract(path, for_rewrite: bool = False) -> Document:
    """for_rewrite=True keeps everything (apparatus as 'protected' blocks, citations
    and inline markdown intact) so the document can be reassembled after rewriting."""
    path = Path(path)
    doc = Document(str(path))
    ext = path.suffix.lower()
    if ext == ".pdf":
        blocks = read_pdf(path, doc)
    elif ext == ".docx":
        blocks = read_docx(path, doc)
    elif ext in (".md", ".markdown"):
        blocks = read_markdown(path.read_text(errors="replace"), doc, keep=for_rewrite)
    else:
        blocks = read_text(path.read_text(errors="replace"), doc)
    if not for_rewrite:
        blocks = [b for b in blocks if b.kind != "protected"]
    blocks = clean_document(doc, blocks, keep=for_rewrite)
    tag_roles(blocks)
    doc.blocks = blocks
    return doc


def extract_text(text: str, name: str = "<text>", fmt: str = "md") -> Document:
    doc = Document(name)
    blocks = read_markdown(text, doc) if fmt == "md" else read_text(text, doc)
    blocks = clean_document(doc, blocks)
    tag_roles(blocks)
    doc.blocks = blocks
    return doc


def log_document(doc: Document) -> None:
    s = doc.summary()
    stripped = ", ".join(f"{k}={v}" for k, v in sorted(s["stripped"].items())) or "nothing"
    log(f"[extract] {Path(doc.path).name}: {s['words']:,} words, {s['paragraphs']} paragraphs; "
        f"stripped: {stripped}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="files or directories")
    ap.add_argument("--text", action="store_true", help="print cleaned text instead of a JSON summary")
    ap.add_argument("--blocks", action="store_true", help="print blocks (with roles) as JSON")
    a = ap.parse_args(argv)
    files = collect_files(a.paths)
    if not files:
        raise SystemExit("no supported files (.pdf .docx .md .txt) found")
    out = []
    for f in files:
        doc = extract(f)
        log_document(doc)
        if a.text:
            print(doc.text, end="\n\n")
        elif a.blocks:
            out.append({"path": doc.path, "blocks": [asdict(b) for b in doc.blocks]})
        else:
            out.append(doc.summary())
    if not a.text:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
