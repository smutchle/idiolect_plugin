"""Deterministic half of the rewrite loop. The model does the writing; this script
does chunking, prompt files, assembly, scoring, gap targeting, stopping rules,
best-version tracking and output.

    rewrite.py prepare  --voice V [--genre G] [--rounds 3] [--output PATH] SOURCE
    rewrite.py assemble RUN_DIR          # after the model has written rewrites/<chunk>.md
    rewrite.py finalize RUN_DIR          # after fidelity; writes the output + scorecard
    rewrite.py status   RUN_DIR

Run directory layout (under ~/.idiolect/runs/):
    state.json, blocks.json, brief.md
    chunks/<cid>.source.md  chunks/<cid>.exemplars.md
    prompts/r1/<cid>.md     prompts/r<k>/<cid>.md (revision prompts)
    rewrites/<cid>.md       <- the model writes here (current version)
    rounds/<k>/{document.md, score.json, rewrites/}
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import idiolect_home, log, read_json, write_json  # noqa: E402
import brief as br  # noqa: E402
import extract as ex  # noqa: E402
import retrieve as rt  # noqa: E402
import tells  # noqa: E402
import score as sc_mod  # noqa: E402
import voiceprofile as vp  # noqa: E402

CHUNK_MIN, CHUNK_MAX = 150, 400
CUT = "[CUT]"           # a rewritten paragraph consisting only of this is removed (--allow-cuts)
N_EXEMPLARS = 4
GAPS_PER_ROUND = 5
MIN_IMPROVEMENT = 0.05

HARD_CONSTRAINTS = """\
- Preserve every claim, number, unit, citation, named entity, and the **strength** of every claim
  ("suggests" must not become "shows", and vice versa; "may" stays "may").
- Do not add or remove arguments, examples, caveats, or qualifications.
- Do not import topic vocabulary, facts, names, or examples from the exemplars. Exemplars are for
  style only: rhythm, syntax, connectives, punctuation, paragraph shape.
- Keep citations, footnote markers, inline code, math, links and markdown formatting exactly as written.
- Keep exactly the same number of paragraphs (and list items) as the source passage, in the same order."""
# (with --allow-cuts a paragraph may be written as [CUT]; it still counts as a paragraph)


# ======================================================================= chunking

def block_md(b: ex.Block) -> str:
    return f"- {b.text}" if b.kind == "list" else b.text


def make_chunks(blocks: list[ex.Block]) -> list[dict]:
    """Group consecutive paragraphs into 150-400 word chunks; never split a paragraph,
    never cross a heading or protected block."""
    chunks, cur, cur_w = [], [], 0

    def close():
        nonlocal cur, cur_w
        if cur:
            chunks.append(cur)
        cur, cur_w = [], 0

    for i, b in enumerate(blocks):
        if b.kind not in ("para", "list"):
            close()
            continue
        w = len(b.text.split())
        if cur and cur_w >= CHUNK_MIN and cur_w + w > CHUNK_MAX:
            close()
        cur.append(i)
        cur_w += w
        if cur_w >= CHUNK_MAX:
            close()
    close()
    # merge a tiny trailing chunk into its predecessor when they are adjacent
    merged = []
    for ch in chunks:
        w = sum(len(blocks[i].text.split()) for i in ch)
        if merged and w < CHUNK_MIN / 2 and merged[-1][-1] + 1 == ch[0] and \
                sum(len(blocks[i].text.split()) for i in merged[-1]) + w <= CHUNK_MAX * 1.25:
            merged[-1] = merged[-1] + ch
        else:
            merged.append(ch)
    out = []
    for k, idx in enumerate(merged, 1):
        roles = [blocks[i].role for i in idx]
        role = max(set(roles), key=roles.count)
        out.append({"id": f"c{k:03d}", "blocks": idx, "role": role, "heading": blocks[idx[0]].heading,
                    "words": sum(len(blocks[i].text.split()) for i in idx), "paragraphs": len(idx)})
    return out


def chunk_text(blocks, ch) -> str:
    return "\n\n".join(block_md(blocks[i]) for i in ch["blocks"])


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


# ======================================================================= documents

def assemble_markdown(blocks: list[ex.Block], replacement: dict[int, str]) -> str:
    out = []
    for i, b in enumerate(blocks):
        if b.kind == "heading":
            out.append("#" * max(1, b.level) + " " + b.text)
        elif b.kind == "protected":
            out.append(b.text)
        else:
            t = replacement.get(i, block_md(b))
            if t.strip() == CUT:
                continue                    # paragraph removed (--allow-cuts)
            if b.kind == "list" and not t.lstrip().startswith(("- ", "* ")):
                t = "- " + t
            out.append(t)
    return "\n\n".join(out) + "\n"


def to_plain(md: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", md, flags=re.M)


def write_docx(src: Path, dest: Path, blocks: list[ex.Block], replacement: dict[int, str]) -> None:
    """Edit a copy of the source .docx in place: styles, tables and images survive.
    Inline formatting inside a rewritten paragraph collapses to its first run's style."""
    import docx
    d = docx.Document(str(src))
    paras = d.paragraphs
    for i, txt in replacement.items():
        b = blocks[i]
        if b.src_index < 0:
            continue
        p = paras[b.src_index]
        if txt.strip() == CUT:
            p._element.getparent().remove(p._element)
            continue
        txt = re.sub(r"^[-*]\s+", "", txt) if b.kind == "list" else txt
        if p.runs:
            p.runs[0].text = txt
            for r in p.runs[1:]:
                r.text = ""
        else:
            p.add_run(txt)
    d.save(str(dest))


def write_output(st: dict, blocks, replacement: dict[int, str], md: str) -> Path:
    out = Path(st["output"])
    fmt = out.suffix.lower().lstrip(".")
    if fmt == "docx" and st["format"] == "docx":
        write_docx(Path(st["source"]), out, blocks, replacement)
    elif fmt == "txt":
        out.write_text(to_plain(md), encoding="utf-8")
    else:
        out.write_text(md, encoding="utf-8")
    return out


# ======================================================================= tells in prompts

def chunk_blocks_with_heading(blocks, ch) -> list:
    """The chunk's blocks, preceded by its heading when it starts a section (for s24)."""
    first = ch["blocks"][0]
    pre = [blocks[first - 1]] if first > 0 and blocks[first - 1].kind == "heading" else []
    return pre + [blocks[i] for i in ch["blocks"]]


WRAPPERS = {p for p, m in tells.PATTERNS.items() if m["removal"] == "wrapper"}


def wrapper_only_paras(blocks, ch) -> list[int]:
    """Positions (0-based, within the chunk) of paragraphs made entirely of chatbot residue or
    staged run-ups: removable with [CUT] even without --allow-cuts (they carry no content)."""
    out = []
    for k, i in enumerate(ch["blocks"]):
        sents = tells._sentences(re.sub(r"\*\*|__", "", blocks[i].text))
        if sents and all(any(h.pattern in WRAPPERS for h in tells.scan_sentences([x])) for x in sents):
            out.append(k)
    return out


def cut_violations(st: dict, ch: dict, paras: list[str]) -> list[int]:
    """[CUT] positions that are not allowed for this run."""
    if st.get("allow_cuts"):
        return []
    ok = set(ch.get("wrapper_only", []))
    return [k for k, x in enumerate(paras) if x.strip() == CUT and k not in ok]


def author_habits(prof: dict | None) -> dict[str, float]:
    """Tell patterns the author really uses (rate per 1k), which must not be 'fixed'."""
    if not prof:
        return {}
    out = {}
    for name, f in prof["features"].items():
        if name.startswith("tell:") and f["count"] >= 10 and f["value"] >= 0.5:
            out[name[5:]] = f["value"]
    return out


def tells_section(hits, words: int, habits: dict[str, float], allow_cuts: bool,
                  wrapper_only: list[int] | None = None) -> str:
    """Prompt section listing the concrete tells found in a passage, with the fix for each."""
    pre = "".join(f"- Paragraph {k + 1} is entirely chatbot residue / run-up: write {CUT} as that paragraph.\n"
                  for k in (wrapper_only or []))
    if not hits:
        return pre + "None detected. Do not introduce any (see the brief's 'Do not use' list)."
    by: dict[str, list] = {}
    for h in hits:
        by.setdefault(h.pattern, []).append(h)
    L = []
    for pid in sorted(by):
        meta = tells.PATTERNS[pid]
        hs = by[pid]
        if pid in habits:
            keep_n = max(0, round(habits[pid] * words / 1000))
            L.append(f"- {pid} {meta['name']}: {len(hs)} here, but this is an author habit "
                     f"({habits[pid]:.1f}/1k): keep about {keep_n} in this passage.")
            continue
        if meta["tier"] == "weak" and len(hits) < 3:
            continue        # weak alone: needs company from other tells
        quotes = "; ".join(f"\"{h.text[:90]}\"" for h in hs[:5]) + (f" (+{len(hs) - 5} more)" if len(hs) > 5 else "")
        if meta["removal"] == "wrapper":
            how = "Remove it; keep any content it wraps."
        elif meta["removal"] == "cut" and allow_cuts:
            how = (f"{meta['fix']} Cutting is allowed here (--allow-cuts) when the removed text carries no fact; "
                   f"to remove a whole paragraph, write {CUT} as that paragraph.")
        else:
            how = (meta["fix"] + " Fidelity first: keep every claim and its strength; change only the form."
                   if meta["removal"] == "cut" else meta["fix"])
        L.append(f"- {pid} {meta['name']} ({meta['tier']}): {quotes}. {how}")
    return pre + ("\n".join(L) if L else "Only isolated weak tells; leave them unless they read as AI.")


# ======================================================================= prompts

def rhythm_target(prof: dict, words: int) -> str:
    bins = br.length_bins(prof)
    mean = prof["features"]["sent_len_mean"]["value"]
    n = max(1, round(words / mean))
    parts = [f"~{round(v * n)} {lab}" for lab, v in bins]
    return (f"For a passage of ~{words} words, the author would write about {n} sentences: "
            + ", ".join(parts) + " (words per sentence).")


def exemplars_md(exs: list[dict]) -> str:
    out = []
    for k, e in enumerate(exs, 1):
        out.append(f"### Exemplar {k} ({e['role']}/{e['function']}, {e['n_words']} words)\n\n{e['text']}")
    return "\n\n".join(out)


def initial_prompt(run: Path, st: dict, ch: dict, src: str) -> str:
    prof = load_profile(st)
    return f"""# Rewrite passage {ch['id']} in the voice of `{st['voice']}`

Role: {ch['role']}. Source: {ch['words']} words in {ch['paragraphs']} paragraph(s).

## Task
Rewrite the SOURCE passage so it reads as if the author wrote it. Study the exemplars for
*how* the author writes this kind of passage: sentence lengths, how sentences open and join,
connectives, punctuation, paragraph shape. Then express the source's exact content that way.

## Style brief
{(run / 'brief.md').read_text(encoding="utf-8").split(chr(10), 2)[2].strip()}

## Rhythm target for this passage
{rhythm_target(prof, ch['words'])}

## Exemplars (style only - never borrow their content, facts, names or topic words)
{(run / 'chunks' / f"{ch['id']}.exemplars.md").read_text(encoding="utf-8").strip()}

## AI tells found in this passage (humanizer catalog; fix every one)
{(run / 'chunks' / f"{ch['id']}.tells.md").read_text(encoding="utf-8").strip()}

## Hard constraints
{HARD_CONSTRAINTS}

## SOURCE
{src}

## Output
Write ONLY the rewritten passage ({ch['paragraphs']} paragraph(s), separated by blank lines;
list items stay as "- " lines) to:
{run / 'rewrites' / (ch['id'] + '.md')}
"""


def sentence_lengths_of(text: str) -> list[int]:
    c = vp.counts_from_text(text)
    return c["sent_lens"]


def revision_prompt(run: Path, st: dict, ch: dict, items: list[str], current: str) -> str:
    src = (run / "chunks" / f"{ch['id']}.source.md").read_text(encoding="utf-8").strip()
    sl = sentence_lengths_of(current)
    return f"""# Revise passage {ch['id']} (round {st['round'] + 1}) - voice `{st['voice']}`

The whole document was scored against the author's profile. This passage contributes most to
the gaps below. Fix these and nothing else; keep everything that already sounds like the author.

## What to change in this passage
{chr(10).join(f"- {x}" for x in items)}

Current sentence lengths in this passage (words): {', '.join(map(str, sl)) or 'n/a'}

## AI tells still present in this passage
{tells_section(tells.scan_text(current)[0], len(current.split()), author_habits(load_profile(st)), st.get('allow_cuts', False))}

## Style brief (for reference)
{(run / 'brief.md').read_text(encoding="utf-8").split(chr(10), 2)[2].strip()}

## Exemplars (style only)
{(run / 'chunks' / f"{ch['id']}.exemplars.md").read_text(encoding="utf-8").strip()}

## Hard constraints
{HARD_CONSTRAINTS}

## ORIGINAL SOURCE (the meaning to preserve)
{src}

## CURRENT VERSION (revise this)
{current.strip()}

## Output
Overwrite {run / 'rewrites' / (ch['id'] + '.md')} with the revised passage
({ch['paragraphs']} paragraph(s), blank-line separated).
"""


# ======================================================================= state

def load_profile(st: dict) -> dict:
    prof, _ = sc_mod.load_voice(st["voice"], st["genre"])
    return prof


def load_blocks(run: Path) -> list[ex.Block]:
    return [ex.Block(**b) for b in read_json(run / "blocks.json")]


def score_markdown(md: str, prof: dict, cal: dict | None, n_gaps: int = 10) -> dict:
    doc = ex.extract_text(md, fmt="md")
    counts = vp.finalize_counts(vp.paragraph_counts(doc.paragraphs), doc.citations, prune=False)
    return sc_mod.score_counts(counts, prof, n_gaps, cal)


def brief_score(s: dict) -> dict:
    out = {k: s.get(k) for k in ("words", "composite", "burrows_delta", "cosine_delta", "inside_band",
                                 "rhythm_inside_band")}
    out["rhythm"] = s["group_scores"].get("rhythm")
    out["band"] = s.get("band")
    out["gaps"] = s["gaps"]
    return out


# ======================================================================= commands

def cmd_prepare(a) -> dict:
    src = Path(a.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    prof, cal = sc_mod.load_voice(a.voice, a.genre)
    doc = ex.extract(src, for_rewrite=True)
    blocks = doc.blocks
    chunks = make_chunks(blocks)
    if not chunks:
        raise SystemExit("no rewritable paragraphs found in the source")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path(a.run_dir) if a.run_dir else idiolect_home() / "runs" / f"{src.stem}-{a.voice}-{a.genre}-{stamp}"
    for d in ("chunks", "prompts/r1", "rewrites", "rounds"):
        (run / d).mkdir(parents=True, exist_ok=True)
    write_json(run / "blocks.json", [b.__dict__ for b in blocks])
    (run / "brief.md").write_text(br.style_brief(prof), encoding="utf-8")
    fmt = src.suffix.lower().lstrip(".")
    out_fmt = {"pdf": "md", "markdown": "md", "text": "txt"}.get(fmt, fmt)
    output = Path(a.output).expanduser().resolve() if a.output else src.with_name(f"{src.stem}.{a.voice}.{out_fmt}")
    st = {"voice": a.voice, "genre": a.genre, "source": str(src), "format": fmt, "output": str(output),
          "max_rounds": a.rounds, "round": 0, "rounds": [], "best_round": None, "done": False,
          "stop_reason": None, "created": stamp, "allow_cuts": bool(getattr(a, "allow_cuts", False)),
          "chunks": chunks}
    # exemplars + prompts
    bank = rt.load_bank(a.voice, a.genre)
    used: list[str] = []
    for ch in chunks:
        text = chunk_text(blocks, ch)
        (run / "chunks" / f"{ch['id']}.source.md").write_text(text + "\n", encoding="utf-8")
        q = rt.describe_query(text, bank.fw_top, role=ch["role"])
        # mild cross-chunk diversity: avoid re-using exemplars of the previous two chunks
        exs = bank.query(q, N_EXEMPLARS, exclude_ids=used[-2 * N_EXEMPLARS:])
        used += [e["id"] for e in exs]
        (run / "chunks" / f"{ch['id']}.exemplars.md").write_text(exemplars_md(exs) + "\n", encoding="utf-8")
        hits = tells.scan_blocks(chunk_blocks_with_heading(blocks, ch))
        ch["wrapper_only"] = wrapper_only_paras(blocks, ch)
        (run / "chunks" / f"{ch['id']}.tells.md").write_text(
            tells_section(hits, ch["words"], author_habits(prof), st["allow_cuts"], ch["wrapper_only"]) + "\n", encoding="utf-8")
        (run / "prompts" / "r1" / f"{ch['id']}.md").write_text(initial_prompt(run, st, ch, text), encoding="utf-8")
    # round 0: the source itself
    src_md = assemble_markdown(blocks, {})
    (run / "rounds" / "0").mkdir(parents=True, exist_ok=True)
    (run / "rounds" / "0" / "document.md").write_text(src_md, encoding="utf-8")
    s0 = score_markdown(src_md, prof, cal)
    write_json(run / "rounds" / "0" / "score.json", s0)
    st["rounds"].append({"round": 0, "kind": "source", **{k: v for k, v in brief_score(s0).items() if k != "gaps"}})
    write_json(run / "state.json", st)
    return {"run_dir": str(run), "chunks": len(chunks), "words": sum(c["words"] for c in chunks),
            "prompts": str(run / "prompts" / "r1"), "rewrites": str(run / "rewrites"),
            "source_score": {"composite": s0["composite"], "rhythm": s0["group_scores"].get("rhythm"),
                             "band": s0.get("band")},
            "output": str(output)}


def chunk_attribution(prof: dict, blocks, chunks, texts: dict[str, str], gaps: list[dict],
                      doc_score: dict) -> dict[str, list[dict]]:
    """For each gap, the chunks that contribute most to it (in the gap's direction)."""
    sel = prof["selection"]
    per_chunk = {}
    for ch in chunks:
        c = vp.counts_from_text(texts[ch["id"]])
        f = vp.compute_features(c, sel["function_words"], [p["phrase"] for p in sel["phrases"]], sel["openers"])
        per_chunk[ch["id"]] = (c, f)
    k_per_gap = max(2, math.ceil(len(chunks) / 3))
    out: dict[str, list[dict]] = {}
    for g in gaps:
        name, sign = g["feature"], (1 if g["value"] > g["target"] else -1)
        contrib = []
        for cid, (c, f) in per_chunk.items():
            if name == "sent_len_w1":
                bins = doc_score["features"]["sent_len_w1"]["bins"]
                worst = max(bins, key=lambda b: abs(b["text"] - b["author"]))
                lo, hi, _ = next(x for x in sc_mod.LEN_BINS if x[2] == worst["label"])
                sl = np.asarray(c["sent_lens"]) if c["sent_lens"] else np.zeros(1)
                share = float(((sl >= lo) & (sl <= hi)).mean())
                d = (share - worst["author"]) * (1 if worst["text"] > worst["author"] else -1) * len(sl)
                detail = f"{100 * share:.0f}% of this passage's sentences are {worst['label']}"
            elif name in f:
                v, n = f[name]["value"], f[name]["n"]
                kind = f[name]["kind"]
                if kind == "rate":
                    d = (v - g["target"]) * n / f[name].get("scale", 1000.0) * sign
                    detail = f"{f[name]['count']} in this passage"
                elif kind == "prop":
                    d = (v - g["target"]) * n * sign
                    detail = f"{100 * v:.0f}% in this passage"
                else:
                    d = (v - g["target"]) * sign * math.sqrt(max(n, 1))
                    detail = f"{v:.1f} in this passage"
            else:
                continue
            if d > 0:
                contrib.append((d, cid, detail))
        contrib.sort(reverse=True)
        for d, cid, detail in contrib[:k_per_gap]:
            out.setdefault(cid, []).append({"feature": name, "instruction": g["instruction"], "detail": detail})
    return out


def cmd_assemble(a) -> dict:
    run = Path(a.run_dir)
    st = read_json(run / "state.json")
    if st["done"]:
        return {"status": "done", "stop_reason": st["stop_reason"], "next": "run finalize"}
    prof, cal = sc_mod.load_voice(st["voice"], st["genre"])
    blocks = load_blocks(run)
    chunks = st["chunks"]
    missing, bad, replacement, texts = [], [], {}, {}
    for ch in chunks:
        p = run / "rewrites" / f"{ch['id']}.md"
        if not p.exists():
            missing.append(ch["id"])
            continue
        paras = split_paragraphs(p.read_text(encoding="utf-8"))
        if len(paras) != ch["paragraphs"]:
            bad.append(f"{ch['id']}: {len(paras)} paragraphs, source has {ch['paragraphs']}")
            continue
        if cut_violations(st, ch, paras):
            bad.append(f"{ch['id']}: {CUT} in paragraph(s) {[k + 1 for k in cut_violations(st, ch, paras)]} - only "
                       f"pure chatbot-residue paragraphs may be cut without --allow-cuts")
            continue
        for i, t in zip(ch["blocks"], paras):
            replacement[i] = t
        texts[ch["id"]] = "\n\n".join(paras)
    if missing or bad:
        return {"status": "incomplete", "missing": missing, "paragraph_mismatch": bad,
                "next": "write/fix these rewrites, then run assemble again"}
    rnd = st["round"] + 1
    rdir = run / "rounds" / str(rnd)
    if rdir.exists():
        shutil.rmtree(rdir)
    (rdir / "rewrites").mkdir(parents=True)
    for ch in chunks:
        shutil.copy(run / "rewrites" / f"{ch['id']}.md", rdir / "rewrites" / f"{ch['id']}.md")
    md = assemble_markdown(blocks, replacement)
    (rdir / "document.md").write_text(md, encoding="utf-8")
    s = score_markdown(md, prof, cal, n_gaps=10)
    write_json(rdir / "score.json", s)
    rec = {"round": rnd, "kind": "rewrite" if rnd == 1 else "revision",
           **{k: v for k, v in brief_score(s).items() if k != "gaps"}}
    st["rounds"].append(rec)
    st["round"] = rnd
    scored = [r for r in st["rounds"] if r["round"] >= 1]
    best = min(scored, key=lambda r: r["composite"])
    st["best_round"] = best["round"]

    # ---- stopping rules
    prev = st["rounds"][-2]["composite"] if rnd >= 2 else None
    inside = bool(s.get("inside_band")) and s.get("rhythm_inside_band", True) is not False
    reason = None
    if inside:
        reason = "inside the within-author band (composite and sentence rhythm)"
    elif rnd - 1 >= st["max_rounds"]:
        reason = f"reached the revision limit ({st['max_rounds']} rounds)"
    elif prev is not None and (prev - s["composite"]) / prev < MIN_IMPROVEMENT:
        reason = (f"round improved distance by less than {100 * MIN_IMPROVEMENT:.0f}% "
                  f"({prev:.2f} -> {s['composite']:.2f})")
    result = {"round": rnd, "composite": s["composite"], "rhythm": s["group_scores"].get("rhythm"),
              "band": s.get("band"), "inside_band": s.get("inside_band"),
              "rhythm_inside_band": s.get("rhythm_inside_band"), "best_round": st["best_round"],
              "history": [(r["round"], round(r["composite"], 3)) for r in st["rounds"]]}
    if reason:
        st["done"] = True
        st["stop_reason"] = reason
        write_json(run / "state.json", st)
        # restore the best version as the current one
        for ch in chunks:
            shutil.copy(run / "rounds" / str(st["best_round"]) / "rewrites" / f"{ch['id']}.md",
                        run / "rewrites" / f"{ch['id']}.md")
        result.update({"status": "stop", "stop_reason": reason,
                       "next": "run the fidelity check (fidelity.py RUN_DIR), fix any flagged passages, "
                               "then finalize"})
        return result

    # ---- revise: build from the best version so far
    base = st["best_round"]
    if base != rnd:
        for ch in chunks:
            shutil.copy(run / "rounds" / str(base) / "rewrites" / f"{ch['id']}.md", run / "rewrites" / f"{ch['id']}.md")
        s = read_json(run / "rounds" / str(base) / "score.json")
        texts = {ch["id"]: (run / "rewrites" / f"{ch['id']}.md").read_text(encoding="utf-8") for ch in chunks}
    gaps = s["gaps"][:GAPS_PER_ROUND]
    targets = chunk_attribution(prof, blocks, chunks, texts, gaps, s)
    pdir = run / "prompts" / f"r{rnd + 1}"
    pdir.mkdir(parents=True, exist_ok=True)
    for cid, items in targets.items():
        ch = next(c for c in chunks if c["id"] == cid)
        lines = [f"{it['instruction']} [{it['detail']}]" for it in items]
        (pdir / f"{cid}.md").write_text(revision_prompt(run, st, ch, lines, texts[cid]), encoding="utf-8")
    write_json(run / "state.json", st)
    result.update({"status": "revise", "gaps": [g["instruction"] for g in gaps],
                   "revise_chunks": sorted(targets), "prompts": str(pdir),
                   "based_on_round": base,
                   "next": "for each prompt file, overwrite rewrites/<chunk>.md as instructed; then assemble"})
    return result


def tells_line(before: Path, after: Path, habits: dict[str, float] | None = None) -> str:
    h0, _ = tells.scan_text(before.read_text(encoding="utf-8"))
    h1, w1 = tells.scan_text(after.read_text(encoding="utf-8"))
    habits = habits or {}
    h0 = [h for h in h0 if h.pattern not in habits]
    h1 = [h for h in h1 if h.pattern not in habits]
    strong = lambda hs: sum(h.pattern in tells.STRONG for h in hs)
    from collections import Counter
    left = Counter(f"{h.pattern} {h.name}" for h in h1)
    rest = ("; remaining: " + ", ".join(f"{k} ({n})" for k, n in left.most_common(6))) if left else ""
    hab = (" Author habits not counted: " + ", ".join(sorted(habits)) + ".") if habits else ""
    return (f"AI tells (humanizer catalog): {len(h0)} -> {len(h1)} "
            f"(strong {strong(h0)} -> {strong(h1)}){rest}.{hab}")


def render_scorecard(run: Path, st: dict) -> str:
    r0 = st["rounds"][0]
    best = next(r for r in st["rounds"] if r["round"] == st["best_round"])
    s_best = read_json(run / "rounds" / str(st["best_round"]) / "score.json")
    b = s_best.get("band") or {}
    fid_path = run / "fidelity.json"
    L = [f"# Scorecard: {Path(st['source']).name} -> voice `{st['voice']}/{st['genre']}`", "",
         f"Output: {st['output']}", f"Stopped: {st['stop_reason']}. Best round: {st['best_round']} "
                                    f"of {st['round']}.", "",
         "| | composite | sentence rhythm | Burrows' Delta | Cosine Delta |", "|---|---|---|---|---|",
         f"| source | {r0['composite']:.2f} | {r0['rhythm'] or float('nan'):.2f} | {r0['burrows_delta']:.2f} | {r0['cosine_delta']:.2f} |",
         f"| output | {best['composite']:.2f} | {best['rhythm'] or float('nan'):.2f} | {best['burrows_delta']:.2f} | {best['cosine_delta']:.2f} |"]
    if b:
        L.append(f"| author band (p10-p90, ~{b['bucket_words']} words) | {b['lo']:.2f}-{b['hi']:.2f} | "
                 f"<= {b.get('rhythm_hi') or float('nan'):.2f} | <= {b.get('delta_hi') or float('nan'):.2f} | |")
        L.append("")
        L.append(f"Composite: **{'INSIDE' if s_best.get('inside_band') else 'OUTSIDE'}** the within-author band. "
                 f"Sentence rhythm: **{'INSIDE' if s_best.get('rhythm_inside_band') else 'OUTSIDE'}**.")
    r = s_best["features"].get("sent_len_w1")
    if r:
        L.append("Sentence lengths (output vs author): " + ", ".join(
            f"{x['label']} {100 * x['text']:.0f}% / {100 * x['author']:.0f}%" for x in r["bins"]))
    L.append("\nRound history: " + " -> ".join(f"r{x['round']} {x['composite']:.2f}" for x in st["rounds"]))
    final_md = run / "final" / "document.md"
    if not final_md.exists():
        final_md = run / "rounds" / str(st["best_round"]) / "document.md"
    L.append(tells_line(run / "rounds" / "0" / "document.md", final_md, author_habits(load_profile(st))))
    if s_best["gaps"]:
        L.append("\n## Remaining top gaps")
        L += [f"- {g['instruction']}" for g in s_best["gaps"][:5]]
    L.append("\n## Fidelity")
    if fid_path.exists():
        fid = read_json(fid_path)
        L.append(fid.get("summary_md", json.dumps(fid, indent=1)))
    else:
        L.append("**NOT RUN.** The fidelity check has not been run on this output. Do not use it until it is.")
    if s_best.get("short_text"):
        L.append("\nNote: the document is short; several features could not be measured reliably.")
    # fail loudly: an unresolved or missing fidelity check goes to the very top
    if not fid_path.exists():
        L.insert(1, "> **FIDELITY NOT CHECKED** - run the fidelity check before using this output.\n")
    elif not read_json(fid_path).get("passed"):
        L.insert(1, "> **FIDELITY FAIL - unresolved meaning changes; see the Fidelity section.**\n")
    return "\n".join(L) + "\n"


def cmd_finalize(a) -> dict:
    run = Path(a.run_dir)
    st = read_json(run / "state.json")
    if st["best_round"] is None:
        raise SystemExit("nothing to finalize: no rewrite round has been assembled")
    blocks = load_blocks(run)
    best_dir = run / "rounds" / str(st["best_round"])
    # fidelity fixes are made in rewrites/ after the loop stops: prefer them when present
    replacement = {}
    src_of = {}
    for ch in st["chunks"]:
        p = run / "rewrites" / f"{ch['id']}.md"
        if not p.exists():
            p = best_dir / "rewrites" / f"{ch['id']}.md"
        paras = split_paragraphs(p.read_text(encoding="utf-8"))
        if len(paras) != ch["paragraphs"]:
            raise SystemExit(f"{ch['id']}: paragraph count changed ({len(paras)} vs {ch['paragraphs']})")
        for i, t in zip(ch["blocks"], paras):
            replacement[i] = t
        src_of[ch["id"]] = str(p)
    md = assemble_markdown(blocks, replacement)
    # re-score the final text (it may include post-loop fidelity fixes)
    prof, cal = sc_mod.load_voice(st["voice"], st["genre"])
    fdir = run / "final"
    fdir.mkdir(exist_ok=True)
    (fdir / "document.md").write_text(md, encoding="utf-8")
    s = score_markdown(md, prof, cal)
    write_json(best_dir / "score.json", s)
    for r in st["rounds"]:
        if r["round"] == st["best_round"]:
            r.update({k: v for k, v in brief_score(s).items() if k != "gaps"})
    out = write_output(st, blocks, replacement, md)
    card = render_scorecard(run, st)
    (run / "scorecard.md").write_text(card, encoding="utf-8")
    card_out = out.with_name(out.stem + ".scorecard.md")
    card_out.write_text(card, encoding="utf-8")
    write_json(run / "state.json", st)
    return {"output": str(out), "scorecard": str(card_out), "composite": s["composite"],
            "inside_band": s.get("inside_band"), "rhythm_inside_band": s.get("rhythm_inside_band"),
            "fidelity_run": (run / "fidelity.json").exists(), "scorecard_md": card}


def cmd_status(a) -> dict:
    st = read_json(Path(a.run_dir) / "state.json")
    return {k: st[k] for k in ("voice", "genre", "source", "output", "round", "best_round", "done", "stop_reason")} | \
        {"history": [(r["round"], round(r["composite"], 3)) for r in st["rounds"]], "chunks": len(st["chunks"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("source")
    p.add_argument("--voice", required=True)
    p.add_argument("--genre", default="general")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output")
    p.add_argument("--run-dir")
    p.add_argument("--allow-cuts", action="store_true",
                   help="let the rewrite cut empty content the humanizer catalog marks as cuttable "
                        "(restating closers, puffery, send-offs, unsupported authority); the fidelity "
                        "reviewer classifies every removal")
    for name in ("assemble", "finalize", "status"):
        q = sub.add_parser(name)
        q.add_argument("run_dir")
    a = ap.parse_args(argv)
    res = {"prepare": cmd_prepare, "assemble": cmd_assemble, "finalize": cmd_finalize, "status": cmd_status}[a.cmd](a)
    card = res.pop("scorecard_md", None) if isinstance(res, dict) else None
    print(json.dumps(res, indent=1, default=str))
    if card:
        print("\n" + card)


if __name__ == "__main__":
    main()
