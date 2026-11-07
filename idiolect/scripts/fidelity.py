"""Fidelity checks: did the rewrite keep the source's meaning?

    fidelity.py check  RUN_DIR                 deterministic check + paragraph pairs for the model pass
    fidelity.py record RUN_DIR MODEL_JSON      merge the model pass, write RUN_DIR/fidelity.json
    fidelity.py compare SOURCE OUTPUT          deterministic check of any two documents

Deterministic part: every number, percentage, unit, citation key and capitalized named
entity in the source must appear in the output, and the output must contain none that
were absent from the source. The model part (claims added / dropped / changed in
strength) is a separate pass with a fresh, narrow prompt - see skills/idiolect/SKILL.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, write_json  # noqa: E402
import extract as ex  # noqa: E402

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000, "half": 0.5, "twice": 2,
    "dozen": 12,
}
UNITS = {
    "%", "percent", "per cent", "percentage points", "pp", "mg", "g", "kg", "lb", "lbs", "oz", "ml", "l",
    "mm", "cm", "m", "km", "mi", "miles", "mile", "ft", "feet", "inch", "inches", "s", "sec", "seconds",
    "second", "min", "minutes", "minute", "h", "hr", "hours", "hour", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "decade", "decades", "°c", "°f", "k", "hz", "khz", "mhz", "ghz",
    "kb", "mb", "gb", "tb", "w", "kw", "mw", "v", "ma", "x", "times", "fold", "dollars", "usd", "eur",
    "participants", "patients", "samples", "subjects", "people", "items", "trials", "studies", "papers",
}
NUM_RE = re.compile(r"(?<![\w.])([$€£]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s?%|\s?(?:percent|per cent)\b)?")
WORD_NUM_RE = re.compile(r"\b(" + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r")\b", re.I)
CITE_PATTERNS = [
    re.compile(r"\[(\d+(?:\s*[-\u2013,]\s*\d+)*)\]"),                     # [3], [1-4], [2, 5]
    re.compile(r"\\cite[pt]?\{([^}]+)\}"),                                # \cite{key}
    re.compile(r"(?<![\w@])@([A-Za-z][\w:-]+\d{2,4}[a-z]?)"),            # pandoc @key2020
    re.compile(r"\[\^([^\]]+)\]"),                                       # footnote [^1]
]
AY_RE = re.compile(r"\b([A-Z][A-Za-z\u00c0-\u017f'\-]+)(?: et al\.?| and [A-Z][A-Za-z'\-]+| & [A-Z][A-Za-z'\-]+)?,?\s*\(?((?:1[6-9]|20)\d\d[a-z]?)\)?")
SENT_START_RE = re.compile(r"(^|[.!?:;]\s+|\n\s*|[\"\u201c(]\s*)$")
STOP_CAPS = {"I", "I'm", "I've", "I'd", "I'll", "OK", "A", "An", "The"}


@lru_cache(maxsize=1)
def nlp_ner():
    import spacy
    return spacy.load("en_core_web_sm")


def _norm_num(sym: str, intpart: str, frac: str | None, pct: str | None) -> str:
    n = intpart.replace(",", "") + (frac or "")
    try:
        v = float(n)
        n = f"{v:g}" if "." in n else str(int(v))
    except ValueError:
        pass
    return f"{sym}{n}{'%' if pct else ''}"


def numbers(text: str) -> set[str]:
    out = {_norm_num(m.group(1), m.group(2), m.group(3), m.group(4)) for m in NUM_RE.finditer(text)}
    for m in WORD_NUM_RE.finditer(text):
        w = m.group(1).lower()
        # 'one' is too often a pronoun ("one of the most..."); count it only before a unit
        if w == "one" and not re.match(r"\s+(" + "|".join(map(re.escape, UNITS)) + r")\b", text[m.end():], re.I):
            continue
        v = NUM_WORDS[w]
        out.add(f"{v:g}" if isinstance(v, float) else str(v))
    return out


def units(text: str) -> set[str]:
    """(number, unit) pairs, e.g. '120 samples', '3 weeks', '4.2%'."""
    out = set()
    # digits may touch their unit ("5mg", "4.2%"); number words need a space ("three weeks")
    num_pat = (r"(?<![\w.])(?:(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?|\b(" + "|".join(NUM_WORDS) + r")\s)")
    for m in re.finditer(num_pat + r"(%|[A-Za-z°]+(?: points| cent)?)(?![A-Za-z])", text, re.I):
        u = m.group(3).lower()
        if u in UNITS:
            n = (m.group(1) or m.group(2)).lower().replace(",", "")
            n = str(NUM_WORDS[n]) if n in NUM_WORDS else n
            u = {"percent": "%", "per cent": "%"}.get(u, u)
            u = u[:-1] if u.endswith("s") and len(u) > 3 and u[:-1] in UNITS else u
            out.add(f"{n} {u}")
    return out


def citations(text: str) -> set[str]:
    out = set()
    for pat in CITE_PATTERNS:
        for m in pat.finditer(text):
            for k in re.split(r"\s*,\s*", m.group(1)):
                out.add(k.strip())
    for m in AY_RE.finditer(text):
        out.add(f"{m.group(1)} {m.group(2)}")
    return out


def capitalized_terms(text: str) -> set[str]:
    """Capitalized words that are not sentence-initial (names, places, products)."""
    out = set()
    # formatting is not content: bold list labels ("**Speed:**"), bold markers, emojis
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])?\s*\W{0,3}\s*\*\*[^*\n]{1,40}:?\*\*:?", " ", text, flags=re.M)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\u2B50\u2705]", " ", text)
    for m in re.finditer(r"\b[A-Z][\w\u00c0-\u017f'\-]*(?:\s+(?:of|de|van|von|the|for)?\s*[A-Z][\w'\-]*)*", text):
        term = m.group(0)
        if term in STOP_CAPS:
            continue
        before = text[max(0, m.start() - 4):m.start()]
        if SENT_START_RE.search(before) or m.start() == 0:
            # sentence-initial: keep only if it is an all-caps acronym or multi-word name
            if not (term.isupper() and len(term) > 1) and " " not in term:
                continue
            first, _, rest = term.partition(" ")
            term = rest if rest and not first.isupper() else term
        out.add(term.strip())
    return {t for t in out if t and t not in STOP_CAPS}


def entities(text: str) -> set[str]:
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])?\s*\W{0,3}\s*\*\*[^*\n]{1,40}:?\*\*:?", " ", text, flags=re.M)
    doc = nlp_ner()(re.sub(r"\*\*|__", "", text))
    keep = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
    return {e.text.strip() for e in doc.ents if e.label_ in keep and any(c.isupper() for c in e.text)}


def _present(term: str, text: str) -> bool:
    def one(t):
        return re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text, re.I) is not None
    if one(term):
        return True
    # NER sometimes joins a list of names into one span ("Wired and The Verge")
    parts = [p for p in re.split(r",\s*|\s+and\s+|\s*&\s*", term) if p.strip()]
    return len(parts) > 1 and all(one(p.strip()) for p in parts)


def compare_texts(src: str, out: str) -> dict:
    res = {}
    for name, fn in (("numbers", numbers), ("units", units), ("citations", citations)):
        a, b = fn(src), fn(out)
        res[name] = {"missing": sorted(a - b), "added": sorted(b - a)}
    # named entities: NER + capitalized-term heuristic, checked by presence in the other text
    ents_src = entities(src) | capitalized_terms(src)
    ents_out = entities(out) | capitalized_terms(out)
    res["entities"] = {
        "missing": sorted(e for e in ents_src if not _present(e, out)),
        "added": sorted(e for e in ents_out if not _present(e, src)),
    }
    res["n_issues"] = sum(len(v["missing"]) + len(v["added"]) for k, v in res.items() if isinstance(v, dict))
    return res


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


# ======================================================================= run dir

def cmd_check(run: Path) -> dict:
    st = read_json(run / "state.json")
    fdir = run / "fidelity"
    (fdir / "pairs").mkdir(parents=True, exist_ok=True)
    per_chunk, pair_files, total = {}, [], 0
    for ch in st["chunks"]:
        cid = ch["id"]
        src = (run / "chunks" / f"{cid}.source.md").read_text()
        rp = run / "rewrites" / f"{cid}.md"
        if not rp.exists():
            raise SystemExit(f"missing rewrite for {cid}")
        out = rp.read_text()
        r = compare_texts(src, "\n\n".join(p for p in split_paragraphs(out) if p.strip() != "[CUT]"))
        if r["n_issues"]:
            per_chunk[cid] = r
            total += r["n_issues"]
        sp, op = split_paragraphs(src), split_paragraphs(out)
        lines = [f"# Chunk {cid}: compare meaning only", ""]
        for k, (a, b) in enumerate(zip(sp, op), 1):
            b = "(paragraph cut)" if b.strip() == "[CUT]" else b
            lines += [f"## Paragraph {k}", "", "SOURCE:", a, "", "REWRITE:", b, ""]
        if len(sp) != len(op):
            lines += ["(paragraph counts differ: compare the remaining text as a whole)", "",
                      "SOURCE (rest):", "\n\n".join(sp[len(op):]), "", "REWRITE (rest):", "\n\n".join(op[len(sp):])]
        pf = fdir / "pairs" / f"{cid}.md"
        pf.write_text("\n".join(lines) + "\n")
        pair_files.append(str(pf))
    det = {"passed": total == 0, "n_issues": total, "chunks": per_chunk}
    write_json(fdir / "deterministic.json", det)
    batches = [pair_files[i:i + 6] for i in range(0, len(pair_files), 6)]
    return {"deterministic": det, "pair_files": pair_files, "model_batches": batches,
            "next": "run the model pass on each batch (fresh subagent, prompt in SKILL.md), save merged "
                    "findings to fidelity/model.json, then: fidelity record RUN_DIR fidelity/model.json"}


def _det_md(det: dict) -> list[str]:
    L = []
    for cid, r in det["chunks"].items():
        for kind in ("numbers", "units", "citations", "entities"):
            v = r[kind]
            if v["missing"]:
                L.append(f"- {cid}: {kind} missing from output: {', '.join(v['missing'])}")
            if v["added"]:
                L.append(f"- {cid}: {kind} added in output: {', '.join(v['added'])}")
    return L


def cmd_record(run: Path, model_json: Path) -> dict:
    det = read_json(run / "fidelity" / "deterministic.json")
    raw = model_json.read_text().strip()
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        model = json.loads(m.group(0) if m else raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"model findings are not valid JSON: {e}")
    if isinstance(model, dict):
        model = model.get("findings", [model])
    # removals of empty content (chatbot wrappers, run-ups; with --allow-cuts also puffery,
    # restating closers, send-offs) are classified 'allowed_cut' by the reviewer: listed, not failures
    allowed = [f for f in model if f.get("type") == "allowed_cut"]
    model = [f for f in model if f.get("type") != "allowed_cut"]
    passed = det["passed"] and not model
    lines = [f"**FIDELITY: {'PASS' if passed else 'FAIL'}** - deterministic issues: {det['n_issues']}; "
             f"meaning differences flagged by the model pass: {len(model)}; intended cuts of empty content: "
             f"{len(allowed)}."]
    lines += _det_md(det)
    for f in allowed:
        lines.append(f"- (allowed cut) {f.get('chunk', '?')} para {f.get('paragraph', '?')}: \"{f.get('source', '')}\"")
    for f in model:
        lines.append(f"- {f.get('chunk', '?')} para {f.get('paragraph', '?')} [{f.get('type', '?')}]: "
                     f"\"{f.get('source', '')}\" -> \"{f.get('rewrite', '')}\". {f.get('explanation', '')}")
    rec = {"passed": passed, "deterministic": det, "model": model, "allowed_cuts": allowed,
           "summary_md": "\n".join(lines)}
    write_json(run / "fidelity.json", rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("run_dir")
    p = sub.add_parser("record")
    p.add_argument("run_dir")
    p.add_argument("model_json")
    p = sub.add_parser("compare")
    p.add_argument("source")
    p.add_argument("output")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        res = cmd_check(Path(a.run_dir))
        det = res["deterministic"]
        print(f"FIDELITY (deterministic): {'PASS' if det['passed'] else 'FAIL'} - {det['n_issues']} issue(s)")
        for line in _det_md(det):
            print(line)
        print(json.dumps({k: v for k, v in res.items() if k != "deterministic"}, indent=1))
    elif a.cmd == "record":
        rec = cmd_record(Path(a.run_dir), Path(a.model_json))
        print(rec["summary_md"])
        if not rec["passed"]:
            sys.exit(1)
    else:
        def text_of(path):
            d = ex.extract(path, for_rewrite=True)
            return "\n\n".join(b.text for b in d.blocks if b.kind in ("para", "list"))
        r = compare_texts(text_of(a.source), text_of(a.output))
        print(json.dumps(r, indent=1))
        if r["n_issues"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
