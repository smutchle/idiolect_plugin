"""Build a stylometric profile of an author from writing samples.

    python voiceprofile.py --voice NAME [--genre GENRE] PATH [PATH ...]

PATHs are files or directories (searched recursively for .pdf/.docx/.md/.txt).
Writes  ~/.idiolect/voices/NAME/GENRE/{profile.json, samples.json, exemplars.jsonl,
sources.txt}  and prints a human-readable summary (or --json).

Design: every sample is reduced to *additive counts* (`Counts`). Features are
computed from counts, so pooled values, per-sample values, leave-one-out profiles
and scores of new text all share one code path.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BOOSTERS, CONTRACTION_RE, DATA_DIR,  # noqa: E402
                    DISCOURSE_WORDS, FUNCTION_POS, HEDGES, LATINATE_SUFFIXES,
                    NOMINAL_SUFFIXES, SUMMARY_OPENERS, TRANSITION_OPENERS,
                    TRANSITION_PHRASES, get_nlp, is_word, log, read_json,
                    syllables, voice_dir, wf_rate_per_1k, write_json, zipf)
import extract as ex  # noqa: E402
import tells  # noqa: E402

N_FUNCTION_WORDS = 150
N_PHRASES = 30
N_OPENERS = 20
SENT_BINS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 10_000]
MIN_SAMPLE_WORDS = 1500         # small files are grouped until a sample reaches this
PSEUDO_SAMPLES = 4              # when fewer than 3 samples, split the corpus

# Reliability thresholds on raw event counts
REL_HIGH, REL_LOW = 30, 10

# 'rhythm' = sentence-length spread & shape (incl. the distribution distance added at
# scoring time); it is weighted highest because it is the most visible voice trait.
# 'paragraphs' (paragraph length) is profiled and reported but weighted 0: a rewrite
# keeps the source's paragraphs, so it could never close that gap.
GROUP_WEIGHTS = {"rhythm": 2.0, "length": 1.0, "paragraphs": 0.0, "function_words": 2.0, "punctuation": 1.0, "syntax": 1.0,
                 "lexical": 0.75, "structure": 0.5, "phrases": 0.75, "ai_tells": 1.0}


# ======================================================================= counts

def empty_counts() -> dict:
    return {
        "words": 0, "sents": 0, "paras": 0, "multi_sent_paras": 0,
        "sent_lens": [], "sent_seqs": [], "para_sents": [], "para_words": [],
        "tok": Counter(), "closed": Counter(), "punct": Counter(),
        "passive": 0, "subj_clauses": 0, "nominal": 0, "fp_sing": 0, "fp_plur": 0,
        "second": 0, "hedge": 0, "booster": 0, "contraction": 0,
        "opener_pos": Counter(), "opener_word": Counter(),
        "depth_sum": 0, "subord": 0, "wordlen_sum": 0, "wordlen_sq": 0,
        "poly3": 0, "latinate": 0, "mattr": [],
        "para_trans_open": 0, "para_summary_close": 0, "list_items": 0,
        "cit_numeric": 0, "cit_ay": 0, "tells": Counter(), "ngrams": Counter(),
    }


def merge_counts(cs: list[dict]) -> dict:
    out = empty_counts()
    for c in cs:
        for k, v in c.items():
            if k.startswith("_"):
                continue
            if isinstance(out[k], Counter):
                out[k].update(v)
            elif isinstance(out[k], list):
                out[k].extend(v)
            else:
                out[k] += v
    return out


def counts_to_json(c: dict) -> dict:
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in c.items() if not k.startswith("_")}


def counts_from_json(d: dict) -> dict:
    c = empty_counts()
    for k, v in d.items():
        if k not in c:          # e.g. counts saved by an older version
            continue
        c[k] = Counter(v) if isinstance(c[k], Counter) else v
    return c


def mattr(tokens: list[str], window: int = 500) -> float:
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    cnt = Counter(tokens[:window])
    total = len(cnt)
    s = total
    for i in range(window, n):
        out_t, in_t = tokens[i - window], tokens[i]
        cnt[out_t] -= 1
        if cnt[out_t] == 0:
            del cnt[out_t]
            total -= 1
        if cnt[in_t] == 0:
            total += 1
        cnt[in_t] += 1
        s += total
    return s / (n - window + 1) / window


def eligible_ngram_token(tok) -> bool:
    """General vocabulary only: never topic words (the topic-leakage guard)."""
    t = tok.lower_
    if not tok.is_alpha or CONTRACTION_RE.match(tok.text):
        return False
    if tok.pos_ in FUNCTION_POS or t in DISCOURSE_WORDS:
        return True
    if tok.pos_ == "PROPN" or tok.pos_ == "NUM":
        return False
    if tok.pos_ == "NOUN":
        return zipf(t) >= 5.2           # 'fact', 'case', 'way', 'point' ...
    return zipf(t) >= 4.5




def paragraph_counts(paras: list[ex.Block]) -> list[dict]:
    """Parse paragraphs with spaCy; return one additive Counts per paragraph.
    Each carries a transient '_words' list so MATTR can be recomputed on merge."""
    nlp = get_nlp()
    out: list[dict] = []
    prev_para_words = 0
    texts = [p.text for p in paras]
    for blk, doc in zip(paras, nlp.pipe(texts, batch_size=64)):
        c = empty_counts()
        all_words: list[str] = []
        c["_words"] = all_words
        out.append(c)
        if blk.kind == "list":
            c["list_items"] += 1
        sents = [s for s in doc.sents if any(is_word(t) for t in s)]
        if not sents:
            continue
        para_words = 0
        seq = []
        for si, s in enumerate(sents):
            words = [t for t in s if is_word(t)]
            n = len(words)
            seq.append(n)
            para_words += n
            c["sent_lens"].append(n)
            # openers
            first = words[0]
            c["opener_pos"][first.pos_] += 1
            nxt = first.nbor(1) if first.i + 1 < len(doc) else None
            key = first.lower_ + ("," if nxt is not None and nxt.text == "," else "")
            if first.pos_ == "PROPN":
                key = "<PROPN>"
            elif first.like_num:
                key = "<NUM>"
            c["opener_word"][key] += 1
            # depth
            root = s.root
            depth = max((len(list(t.ancestors)) for t in s), default=0) if len(s) < 300 else 0
            c["depth_sum"] += depth
            for t in s:
                d = t.dep_
                if d in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl"):
                    c["subj_clauses"] += 1
                if d in ("nsubjpass", "csubjpass"):
                    c["passive"] += 1
                if d in ("advcl", "ccomp", "relcl", "acl", "csubj", "csubjpass"):
                    c["subord"] += 1
            del root
        # AI tells (humanizer catalog; see tells.py)
        stexts = [s.text.strip() for s in sents]
        tags = [[t.pos_ for t in s if not t.is_punct and not t.is_space] for s in sents]
        for h in tells.scan_sentences(stexts, -1, tags):
            c["tells"][h.pattern] += 1
        if prev_para_words >= 30 and len(sents) == 1 and len(stexts[0].split()) <= 10 and blk.kind == "para":
            c["tells"]["s02"] += 1          # one-line closer paragraph
        prev_para_words = sum(1 for t in doc if is_word(t))
        c["sent_seqs"].append(seq)
        c["para_sents"].append(len(sents))
        c["para_words"].append(para_words)
        c["paras"] += 1
        # paragraph opening / closing patterns
        ptxt = blk.text.strip().lower()
        first_tok = next((t for t in sents[0] if is_word(t)), None)
        if first_tok is not None and (
                first_tok.lower_ in TRANSITION_OPENERS
                or any(ptxt.startswith(ph) for ph in TRANSITION_PHRASES)
                or (first_tok.pos_ == "ADV" and first_tok.i + 1 < len(doc) and doc[first_tok.i + 1].text == ",")):
            c["para_trans_open"] += 1
        if len(sents) >= 2:
            c["multi_sent_paras"] += 1
            last = sents[-1].text.strip().lower()
            if last.startswith(SUMMARY_OPENERS):
                c["para_summary_close"] += 1
        # tokens
        prev_eligible: list[str] = []
        for t in doc:
            if t.is_sent_start:
                prev_eligible = []
            if t.is_punct or t.is_space:
                txt = t.text
                if txt == ";":
                    c["punct"]["semicolon"] += 1
                elif txt == ":":
                    c["punct"]["colon"] += 1
                elif txt == ",":
                    c["punct"]["comma"] += 1
                elif txt == "?":
                    c["punct"]["question"] += 1
                elif txt == "!":
                    c["punct"]["exclam"] += 1
                elif txt == "(":
                    c["punct"]["paren"] += 1
                elif txt in ('"', "“", "”"):
                    c["punct"]["quote"] += 1
                if "—" in txt or txt == "--":
                    c["punct"]["dash"] += 1
                    c["punct"]["em_dash"] += 1
                elif txt == "–" or (txt == "-" and t.whitespace_ and t.i > 0 and doc[t.i - 1].whitespace_):
                    c["punct"]["dash"] += 1
                prev_eligible = [] if txt not in ("'", "’") else prev_eligible
                continue
            if not is_word(t):
                prev_eligible = []
                continue
            low = t.lower_
            c["words"] += 1
            c["tok"][low] += 1
            all_words.append(low)
            if t.pos_ in FUNCTION_POS:
                c["closed"][low] += 1
            if CONTRACTION_RE.match(t.text) and (t.pos_ in ("AUX", "PART", "VERB") or low.endswith("t")):
                c["contraction"] += 1
            if t.pos_ == "NOUN" and len(low) >= 7 and low.endswith(NOMINAL_SUFFIXES):
                c["nominal"] += 1
            if low in ("i", "me", "my", "mine", "myself"):
                c["fp_sing"] += 1 if not (low == "i" and t.text == "I" and t.pos_ == "NUM") else 0
            elif low in ("we", "us", "our", "ours", "ourselves"):
                c["fp_plur"] += 1
            elif low in ("you", "your", "yours", "yourself", "yourselves"):
                c["second"] += 1
            if low in HEDGES:
                c["hedge"] += 1
            if low in BOOSTERS:
                c["booster"] += 1
            alpha = re.sub(r"[^a-z]", "", low)
            c["wordlen_sum"] += len(alpha)
            c["wordlen_sq"] += len(alpha) ** 2
            if syllables(alpha) >= 3:
                c["poly3"] += 1
            if len(alpha) >= 6 and alpha.endswith(LATINATE_SUFFIXES):
                c["latinate"] += 1
            # signature n-grams (general vocabulary only)
            if eligible_ngram_token(t):
                prev_eligible.append(low)
                prev_eligible = prev_eligible[-4:]
                for n in (2, 3, 4):
                    if len(prev_eligible) >= n:
                        c["ngrams"][" ".join(prev_eligible[-n:])] += 1
            else:
                prev_eligible = []
        c["sents"] += len(sents)
    return out


def finalize_counts(pcs: list[dict], citations: Counter | None = None, prune: bool = True) -> dict:
    """Merge paragraph counts into one Counts (MATTR recomputed over the joined text)."""
    c = merge_counts(pcs)
    words = [w for p in pcs for w in p.get("_words", [])]
    c["mattr"] = [[mattr(words), len(words)]] if words else []
    if citations:
        c["cit_numeric"] += citations.get("numeric", 0)
        c["cit_ay"] += citations.get("author_year", 0)
    if prune:  # drop hapax n-grams to keep stored counts small
        c["ngrams"] = Counter({k: v for k, v in c["ngrams"].items() if v >= 2})
    return c


def counts_from_paragraphs(paras: list[ex.Block], citations: Counter | None = None) -> dict:
    """Parse paragraphs with spaCy and reduce them to additive counts."""
    return finalize_counts(paragraph_counts(paras), citations)


def counts_from_document(doc: ex.Document) -> dict:
    return counts_from_paragraphs(doc.paragraphs, doc.citations)


def counts_from_text(text: str, fmt: str = "md") -> dict:
    return counts_from_document(ex.extract_text(text, fmt=fmt))


# ======================================================================= features

def _q(a, p):
    return float(np.percentile(a, p)) if len(a) else 0.0


def autocorr(seqs: list[list[int]]) -> float:
    flat = [x for s in seqs for x in s]
    if len(flat) < 3:
        return 0.0
    mu = float(np.mean(flat))
    var = float(np.mean([(x - mu) ** 2 for x in flat]))
    if var == 0:
        return 0.0
    pairs = [(s[i] - mu) * (s[i + 1] - mu) for s in seqs for i in range(len(s) - 1)]
    return float(np.mean(pairs) / var) if pairs else 0.0


def compute_features(c: dict, fw_list: list[str], phrases: list[str], openers: list[str]) -> dict:
    """Return {name: {value, group, kind, n, count, scale?, unit_sd?}}.

    kind: 'rate'  -> value = scale * count / n   (Poisson noise: var = scale*value/n)
          'prop'  -> value = count / n          (binomial noise)
          'mean'  -> mean of n units with unit_sd
          'stat'  -> other statistic over n units (noise scaled from between-sample SD)
    """
    W = max(1, c["words"])
    S = max(1, c["sents"])
    P = max(1, c["paras"])
    sl = np.array(c["sent_lens"], dtype=float)
    F: dict[str, dict] = {}

    def rate(name, group, count, n=W, scale=1000.0):
        F[name] = {"value": scale * count / max(1, n), "group": group, "kind": "rate",
                   "n": n, "count": count, "scale": scale}

    def prop(name, group, count, n):
        F[name] = {"value": count / max(1, n), "group": group, "kind": "prop", "n": n, "count": count}

    def stat(name, group, value, n, kind="stat", unit_sd=None):
        F[name] = {"value": float(value), "group": group, "kind": kind, "n": n, "count": n}
        if unit_sd is not None:
            F[name]["unit_sd"] = float(unit_sd)

    # ---- length & rhythm
    sd = float(sl.std(ddof=1)) if len(sl) > 1 else 0.0
    stat("sent_len_mean", "length", sl.mean() if len(sl) else 0, len(sl), "mean", sd)
    stat("sent_len_sd", "rhythm", sd, len(sl))
    lsl = np.log(sl) if len(sl) else sl
    stat("sent_loglen_sd", "rhythm", float(lsl.std(ddof=1)) if len(sl) > 1 else 0.0, len(sl))
    for p in (10, 25, 50, 75, 90):
        stat(f"sent_len_p{p}", "length", _q(sl, p), len(sl))
    prop("sent_short_share", "rhythm", int((sl < 10).sum()), len(sl))
    prop("sent_long_share", "rhythm", int((sl > 35).sum()), len(sl))
    hist, _ = np.histogram(sl, bins=SENT_BINS)
    for lo, hi, h in zip(SENT_BINS[:-1], SENT_BINS[1:], hist):
        prop(f"sent_hist_{lo}_{hi if hi < 10000 else 'inf'}", "length_hist", int(h), len(sl))
    stat("sent_len_autocorr", "rhythm", autocorr(c["sent_seqs"]), len(sl))
    ps = np.array(c["para_sents"], dtype=float)
    pw = np.array(c["para_words"], dtype=float)
    stat("para_len_sents_mean", "paragraphs", ps.mean() if len(ps) else 0, len(ps), "mean",
         ps.std(ddof=1) if len(ps) > 1 else 0)
    stat("para_len_sents_sd", "paragraphs", ps.std(ddof=1) if len(ps) > 1 else 0, len(ps))
    stat("para_len_words_mean", "paragraphs", pw.mean() if len(pw) else 0, len(pw), "mean",
         pw.std(ddof=1) if len(pw) > 1 else 0)
    stat("para_len_words_sd", "paragraphs", pw.std(ddof=1) if len(pw) > 1 else 0, len(pw))

    # positive scale-type statistics vary multiplicatively -> compare them in log space
    for k in ("sent_len_mean", "sent_len_sd", "sent_len_p10", "sent_len_p25", "sent_len_p50",
              "sent_len_p75", "sent_len_p90", "para_len_sents_mean", "para_len_sents_sd",
              "para_len_words_mean", "para_len_words_sd"):
        F[k]["log"] = True

    # ---- function words
    for w in fw_list:
        rate(f"fw:{w}", "function_words", c["tok"].get(w, 0))

    # ---- punctuation
    for k in ("semicolon", "colon", "dash", "paren", "question", "exclam", "quote"):
        rate(f"punct_{k}", "punctuation", c["punct"].get(k, 0))
    rate("commas_per_sent", "punctuation", c["punct"].get("comma", 0), S, scale=1.0)

    # ---- syntax & stance
    prop("passive_share", "syntax", c["passive"], c["subj_clauses"])
    rate("nominalization_rate", "syntax", c["nominal"])
    rate("first_person_sing_rate", "syntax", c["fp_sing"])
    rate("first_person_plur_rate", "syntax", c["fp_plur"])
    rate("second_person_rate", "syntax", c["second"])
    rate("hedge_rate", "syntax", c["hedge"])
    rate("booster_rate", "syntax", c["booster"])
    rate("contraction_rate", "syntax", c["contraction"])
    stat("dep_depth_mean", "syntax", c["depth_sum"] / S, S)
    rate("subord_per_sent", "syntax", c["subord"], S, scale=1.0)
    for pos in ("PRON", "DET", "ADP", "ADV", "SCONJ", "CCONJ", "NOUN", "PROPN", "VERB", "ADJ", "AUX"):
        prop(f"opener_pos:{pos}", "syntax", c["opener_pos"].get(pos, 0), S)
    for o in openers:
        prop(f"opener:{o}", "syntax", c["opener_word"].get(o, 0), S)

    # ---- lexical texture
    m_pairs = c["mattr"]
    m_n = sum(n for _, n in m_pairs)
    stat("mattr_500", "lexical", sum(v * n for v, n in m_pairs) / max(1, m_n), W)
    wl_mean = c["wordlen_sum"] / W
    wl_sd = math.sqrt(max(0.0, c["wordlen_sq"] / W - wl_mean ** 2))
    stat("word_len_mean", "lexical", wl_mean, W, "mean", wl_sd)
    prop("polysyllabic_share", "lexical", c["poly3"], W)
    prop("latinate_share", "lexical", c["latinate"], W)

    # ---- structure
    prop("para_transition_open_share", "structure", c["para_trans_open"], P)
    prop("para_summary_close_share", "structure", c["para_summary_close"], max(1, c["multi_sent_paras"]))
    rate("list_item_rate", "structure", c["list_items"])
    rate("citation_numeric_rate", "structure", c["cit_numeric"])
    rate("citation_author_year_rate", "structure", c["cit_ay"])

    # ---- signature phrases
    for ph in phrases:
        rate(f"phrase:{ph}", "phrases", c["ngrams"].get(ph, 0))

    # ---- AI tells (scored one-sided: only an excess over the author counts)
    # (humanizer catalog, tells.py; formatting/heading patterns are not visible in
    # profile-mode text, so they are reported by the rewrite tools but not profiled)
    for pid in tells.PROSE_PATTERNS:
        rate(f"tell:{pid}", "ai_tells", c["tells"].get(pid, 0))
    return F


# ======================================================================= selection

def select_function_words(c: dict, k: int = N_FUNCTION_WORDS) -> list[str]:
    cand = Counter({w: n for w, n in c["closed"].items() if w.isalpha() or "'" in w})
    # closed-class tokens tagged inconsistently by spaCy (e.g. 'that') count once
    for w in DISCOURSE_WORDS:
        cand[w] = max(cand.get(w, 0), c["tok"].get(w, 0))
    top = [w for w, _ in cand.most_common(k)]
    # always track the discourse list so that *avoidance* (e.g. never 'moreover')
    # is part of the profile
    return top + sorted(w for w in DISCOURSE_WORDS if w not in top)


def select_phrases(c: dict, sample_counts: list[dict], k: int = N_PHRASES) -> list[dict]:
    W = max(1, c["words"])
    min_count = max(4, int(W / 8000))
    scored = []
    for ph, n in c["ngrams"].items():
        if n < min_count:
            continue
        toks = ph.split()
        # pure high-frequency grammar ('of the', 'it is') is covered by function words
        if all(zipf(t) >= 6.3 for t in toks):
            continue
        spread = sum(1 for sc in sample_counts if sc["ngrams"].get(ph, 0) > 0)
        if len(sample_counts) >= 2 and spread < max(2, math.ceil(0.4 * len(sample_counts))):
            continue
        p_obs = n / W
        log_exp = sum(math.log(max(wf_rate_per_1k(t) / 1000.0, 1e-8)) for t in toks)
        pmi = (math.log(p_obs) - log_exp) / len(toks)       # per-token log ratio vs. generic English
        scored.append({"phrase": ph, "count": n, "rate_per_1k": 1000 * p_obs,
                       "score": pmi * math.log(n), "spread": spread})
    scored.sort(key=lambda d: -d["score"])
    # drop phrases subsumed by a higher-ranked longer phrase with near-equal count
    kept: list[dict] = []
    for d in scored:
        if any((d["phrase"] in k_["phrase"] or k_["phrase"] in d["phrase"]) and
               abs(k_["count"] - d["count"]) <= 0.25 * max(k_["count"], d["count"]) for k_ in kept):
            continue
        kept.append(d)
        if len(kept) >= k:
            break
    return kept


def select_openers(c: dict, k: int = N_OPENERS) -> list[str]:
    return [w for w, _ in c["opener_word"].most_common(k)]


# ======================================================================= profile

def reliability(f: dict, name: str, corpus_words: int) -> tuple[str, float]:
    if f["kind"] in ("stat", "mean"):
        cnt = f["n"]
    elif f["kind"] == "prop":
        cnt = min(f["count"], f["n"] - f["count"]) if f["n"] else 0
        cnt = max(cnt, f["count"])  # a share seen often enough is measurable
    else:
        cnt = f["count"]
    # absence of a common discourse word over a big corpus is informative
    if name.startswith("fw:") and cnt < REL_LOW and corpus_words >= 20_000:
        expected = wf_rate_per_1k(name[3:]) * corpus_words / 1000
        if expected >= REL_LOW:
            return "medium", 0.5
    if cnt >= REL_HIGH:
        return "high", 1.0
    if cnt >= REL_LOW:
        return "medium", 0.5
    return "low", 0.0


def robust_sd(vals: list[float]) -> float:
    """Between-sample spread that one unusual sample cannot blow up: 1.4826*MAD,
    floored at half the classical SD (MAD is unstable with very few samples)."""
    if len(vals) < 2:
        return 0.0
    a = np.asarray(vals, dtype=float)
    sd = float(a.std(ddof=1))
    if len(a) < 5:
        return sd
    mad = 1.4826 * float(np.median(np.abs(a - np.median(a))))
    return min(sd, max(mad, 0.5 * sd))


def build_profile(sample_counts: list[dict], meta: dict | None = None,
                  selection: dict | None = None) -> dict:
    """Aggregate per-sample counts into a profile. `selection` (fw/phrases/openers)
    can be fixed from outside, e.g. for leave-one-out calibration."""
    pooled = merge_counts(sample_counts)
    if selection is None:
        selection = {
            "function_words": select_function_words(pooled),
            "phrases": select_phrases(pooled, sample_counts),
            "openers": select_openers(pooled),
        }
    phrase_list = [p["phrase"] for p in selection["phrases"]]
    args = (selection["function_words"], phrase_list, selection["openers"])
    pooled_f = compute_features(pooled, *args)
    per = [compute_features(sc, *args) for sc in sample_counts]
    feats = {}
    for name, f in pooled_f.items():
        vals = [p[name]["value"] for p in per]
        between = robust_sd(vals)
        rel, w = reliability(f, name, pooled["words"])
        entry = {"value": f["value"], "group": f["group"], "kind": f["kind"],
                 "count": f["count"], "n": f["n"], "per_sample": vals,
                 "per_sample_n": [p[name]["n"] for p in per],
                 "between_sd": between, "reliability": rel, "weight": w}
        for k in ("scale", "unit_sd", "log"):
            if k in f:
                entry[k] = f[k]
        if f.get("log") and all(v > 0 for v in vals):
            entry["between_sd_log"] = robust_sd([math.log(v) for v in vals])
        feats[name] = entry
    prof = {
        "meta": dict(meta or {}),
        "corpus": {"words": pooled["words"], "sentences": pooled["sents"],
                   "paragraphs": pooled["paras"], "samples": len(sample_counts),
                   "sample_words": [sc["words"] for sc in sample_counts]},
        "selection": selection,
        "opener_distribution": {k: v / max(1, pooled["sents"]) for k, v in
                                pooled["opener_word"].most_common(N_OPENERS)},
        "group_weights": GROUP_WEIGHTS,
        # paragraph-level sentence-length sequences per sample: the null model for the
        # sentence-length distribution distance (score.rhythm_w1)
        "sentence_lengths": [sc["sent_seqs"] for sc in sample_counts],
        "features": feats,
    }
    prof["warnings"] = profile_warnings(prof)
    return prof


def expected_noise_sd(f: dict, value: float, n: float) -> float:
    """Sampling noise of a feature measured on n units, given the true value."""
    n = max(1.0, float(n))
    k = f["kind"]
    if k == "rate":
        s = f.get("scale", 1000.0)
        return math.sqrt(s * max(value, s * 0.5 / n) / n)
    if k == "prop":
        p = min(max(value, 0.5 / n), 1 - 0.5 / n)
        return math.sqrt(p * (1 - p) / n)
    if k == "mean":
        return f.get("unit_sd", 0.0) / math.sqrt(n)
    # stat: scale the observed between-sample spread by sample size
    ns = np.mean(f.get("per_sample_n") or [n])
    return f.get("between_sd", 0.0) * math.sqrt(max(1.0, ns / n))


MAJOR_FEATURES = ["sent_len_mean", "sent_len_sd", "passive_share", "nominalization_rate",
                  "first_person_sing_rate", "first_person_plur_rate", "second_person_rate",
                  "contraction_rate", "word_len_mean", "punct_semicolon", "commas_per_sent",
                  "para_len_words_mean"]


def profile_warnings(prof: dict) -> list[str]:
    warns = []
    corp = prof["corpus"]
    if corp["words"] < 5000:
        warns.append(f"Corpus is only {corp['words']:,} words; most features will be unreliable "
                     "(>=20k words per genre is recommended).")
    elif corp["words"] < 20000:
        warns.append(f"Corpus is {corp['words']:,} words; >=20k words per genre is recommended "
                     "for stable function-word and punctuation targets.")
    feats = prof["features"]
    if corp["samples"] >= 3:
        mixed, outliers = [], Counter()
        for name in MAJOR_FEATURES:
            f = feats.get(name)
            if not f or f["reliability"] == "low":
                continue
            ns = float(np.mean(f["per_sample_n"]))
            noise = expected_noise_sd(f, f["value"], ns) if f["kind"] != "stat" else 0
            if noise <= 0:
                continue
            ratio = f["between_sd"] / noise
            if ratio > 4.0:
                mixed.append(f"{name} (spread {ratio:.1f}x sampling noise)")
                vals = np.array(f["per_sample"])
                outliers[int(np.argmax(np.abs(vals - f["value"])))] += 1
        if len(mixed) >= 3:
            srcs = prof["meta"].get("samples", [])
            who = ", ".join(srcs[i] if i < len(srcs) else f"sample {i}" for i, _ in outliers.most_common(3))
            warns.append("Samples may mix genres: high between-sample variance on " + "; ".join(mixed[:5])
                         + f". Most deviant samples: {who}. Consider profiling them as a separate --genre.")
    low = sorted(n for n, f in feats.items() if f["reliability"] == "low"
                 and f["group"] not in ("ai_tells", "length_hist", "function_words")
                 and not n.startswith("opener") and not n.startswith("citation"))
    low_fw = sum(1 for n, f in feats.items() if f["reliability"] == "low" and n.startswith("fw:"))
    if low:
        warns.append(f"Low-reliability features (too few occurrences; excluded from targets): "
                     f"{', '.join(low[:10])}{' ...' if len(low) > 10 else ''}.")
    if low_fw:
        warns.append(f"{low_fw} tracked function/discourse words are too rare in the corpus to set targets.")
    return warns


# ======================================================================= samples

def make_samples(docs: list[ex.Document]) -> tuple[list[list[ex.Block]], list[str], list[Counter], bool]:
    """Group tiny files into samples of >= MIN_SAMPLE_WORDS; split into pseudo-samples
    when there are fewer than 3, so between-sample variance can be estimated."""
    groups, names, cits = [], [], []
    cur, cur_names, cur_cit, cur_w = [], [], Counter(), 0
    for d in docs:
        cur.extend(d.paragraphs)
        cur_names.append(Path(d.path).name)
        cur_cit.update(d.citations)
        cur_w += d.n_words
        if cur_w >= MIN_SAMPLE_WORDS:
            groups.append(cur); names.append("+".join(cur_names)); cits.append(cur_cit)
            cur, cur_names, cur_cit, cur_w = [], [], Counter(), 0
    if cur:
        if groups and cur_w < MIN_SAMPLE_WORDS / 2:
            groups[-1].extend(cur); names[-1] += "+" + "+".join(cur_names); cits[-1].update(cur_cit)
        else:
            groups.append(cur); names.append("+".join(cur_names)); cits.append(cur_cit)
    pseudo = False
    if len(groups) < 3:
        pseudo = True
        paras = [p for g in groups for p in g]
        total = sum(len(p.text.split()) for p in paras)
        k = PSEUDO_SAMPLES if total >= 4 * 600 else 3
        target = total / k
        groups, names, acc, cur = [], [], 0, []
        for p in paras:
            cur.append(p)
            acc += len(p.text.split())
            if acc >= target * (len(groups) + 1) and len(groups) < k - 1:
                groups.append(cur); names.append(f"part{len(groups)}"); cur = []
        if cur:
            groups.append(cur); names.append(f"part{len(groups) + 1}")
        total_cit = Counter()
        for cc in cits:
            total_cit.update(cc)
        # spread citations proportionally (they are only used as rates)
        cits = [Counter({k_: v * sum(len(p.text.split()) for p in g) / max(1, total)
                         for k_, v in total_cit.items()}) for g in groups]
    return groups, names, cits, pseudo


# ======================================================================= exemplars

def mask_sequence(doc) -> str:
    """Function words kept, content words replaced by POS: topic-free structure."""
    out = []
    for t in doc:
        if t.is_space:
            continue
        if t.is_punct:
            out.append(t.text if t.text in ",;:?!()—-" else "PUNCT")
        elif t.pos_ in FUNCTION_POS or t.lower_ in DISCOURSE_WORDS:
            out.append(t.lower_)
        else:
            out.append(t.pos_)
    return " ".join(out)


def paragraph_function(doc, text: str) -> str:
    low = text.lower()
    words = [t for t in doc if is_word(t)]
    n = len(words)
    first = words[0].lower_ if words else ""
    if n <= 90 and (first in TRANSITION_OPENERS or low.startswith(TRANSITION_PHRASES)):
        return "transition"
    past = sum(1 for t in doc if t.tag_ in ("VBD", "VBN"))
    passive = sum(1 for t in doc if t.dep_ in ("nsubjpass", "auxpass"))
    stance = sum(1 for t in words if t.lower_ in HEDGES or t.lower_ in BOOSTERS
                 or t.lower_ in ("but", "however", "yet", "because", "therefore", "thus", "must", "should"))
    if re.search(r"\b(we|i) (used|collected|measured|trained|computed|ran|selected)\b", low) or \
            (past + passive) / max(1, n) > 0.12 and passive >= 2:
        return "method"
    if stance / max(1, n) > 0.03:
        return "argument"
    return "description"


def exemplar_features(doc, text: str, fw_top: list[str]) -> dict:
    """Topic-free structural description of a passage (shared by the exemplar bank
    and by retrieval queries)."""
    words = [t for t in doc if is_word(t)]
    sents = [s for s in doc.sents if any(is_word(t) for t in s)]
    sl = [sum(1 for t in s if is_word(t)) for s in sents]
    W = max(1, len(words))
    tok = Counter(t.lower_ for t in words)
    punct = Counter(t.text for t in doc if t.is_punct)
    return {
        "function": paragraph_function(doc, text),
        "n_words": len(words), "n_sents": len(sents),
        "sent_len_mean": float(np.mean(sl)) if sl else 0.0,
        "sent_len_sd": float(np.std(sl)) if len(sl) > 1 else 0.0,
        "fw": [round(1000 * tok.get(w, 0) / W, 2) for w in fw_top],
        "punct": {k: round(1000 * punct.get(ch, 0) / W, 2) for k, ch in
                  (("semicolon", ";"), ("colon", ":"), ("comma", ","), ("question", "?"),
                   ("paren", "("), ("dash", "\u2014"))},
        "masked": mask_sequence(doc),
    }


EXEMPLAR_FW = 60
EXEMPLAR_MIN_WORDS = 40


def build_exemplars(groups: list[list[ex.Block]], names: list[str], fw_top: list[str]) -> list[dict]:
    nlp = get_nlp()
    recs = []
    for gi, (g, src) in enumerate(zip(groups, names)):
        paras = [p for p in g if p.kind == "para" and len(p.text.split()) >= EXEMPLAR_MIN_WORDS]
        for pi, (blk, doc) in enumerate(zip(paras, nlp.pipe([p.text for p in paras], batch_size=64))):
            if ex.is_equation(blk.text) or blk.text.count("|") > 3:
                continue
            rec = {"id": f"{gi}:{pi}", "source": src, "role": blk.role, "heading": blk.heading}
            rec.update(exemplar_features(doc, blk.text, fw_top))
            rec["text"] = blk.text
            recs.append(rec)
    return recs


# ======================================================================= summary

_REF_CACHE = DATA_DIR / "reference_features.json"


def reference_features(selection: dict) -> dict:
    """Features of the bundled generic/AI-style reference text (for contrast)."""
    ref_path = DATA_DIR / "reference_generic.md"
    if not ref_path.exists():
        return {}
    c = counts_from_text(ref_path.read_text(encoding="utf-8"))
    return compute_features(c, selection["function_words"],
                            [p["phrase"] for p in selection["phrases"]], selection["openers"])


PLAIN = {
    "sent_len_mean": "average sentence length (words)",
    "sent_len_sd": "sentence-length variation (SD)",
    "sent_loglen_sd": "sentence-length spread (SD of log length)",
    "sent_short_share": "share of short sentences (<10 words)",
    "sent_long_share": "share of long sentences (>35 words)",
    "sent_len_autocorr": "long/short alternation (lag-1 autocorrelation; negative = alternates)",
    "para_len_words_mean": "paragraph length (words)",
    "para_len_sents_mean": "paragraph length (sentences)",
    "punct_semicolon": "semicolons per 1k words",
    "punct_colon": "colons per 1k words",
    "punct_dash": "dashes per 1k words",
    "punct_paren": "parentheses per 1k words",
    "punct_question": "questions per 1k words",
    "punct_exclam": "exclamation marks per 1k words",
    "punct_quote": "quotation marks per 1k words",
    "commas_per_sent": "commas per sentence",
    "passive_share": "passive clauses (share of clauses)",
    "nominalization_rate": "nominalizations per 1k words",
    "first_person_sing_rate": "I/me/my per 1k words",
    "first_person_plur_rate": "we/us/our per 1k words",
    "second_person_rate": "you/your per 1k words",
    "hedge_rate": "hedges per 1k words",
    "booster_rate": "boosters per 1k words",
    "contraction_rate": "contractions per 1k words",
    "dep_depth_mean": "syntactic depth",
    "subord_per_sent": "subordinate clauses per sentence",
    "mattr_500": "vocabulary variety (MATTR)",
    "word_len_mean": "mean word length",
    "polysyllabic_share": "share of 3+ syllable words",
    "latinate_share": "Latinate-suffix word share",
    "para_transition_open_share": "paragraphs opening with a transition",
    "para_summary_close_share": "paragraphs closing with a summary sentence",
    "para_len_sents_sd": "variation in paragraph length (sentences, SD)",
    "para_len_words_sd": "variation in paragraph length (words, SD)",
    "sent_len_p10": "10th-percentile sentence length",
    "sent_len_p25": "25th-percentile sentence length",
    "sent_len_p50": "median sentence length",
    "sent_len_p75": "75th-percentile sentence length",
    "sent_len_p90": "90th-percentile sentence length",
}
POS_PLAIN = {"PRON": "a pronoun", "DET": "a determiner (the/this/a)", "ADP": "a preposition",
             "ADV": "an adverb", "SCONJ": "a subordinator (when/if/because)", "CCONJ": "and/but/or",
             "NOUN": "a noun", "PROPN": "a proper noun", "VERB": "a verb", "ADJ": "an adjective",
             "AUX": "an auxiliary verb"}
for _pos, _txt in POS_PLAIN.items():
    PLAIN[f"opener_pos:{_pos}"] = f"sentences opening with {_txt}"


def fmt_val(name: str, v: float) -> str:
    if name.endswith("_share") or name.startswith("opener"):
        return f"{100 * v:.0f}%"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def distinctive_habits(prof: dict, ref: dict, k: int = 8) -> list[str]:
    feats = prof["features"]
    cands = []
    for name, f in feats.items():
        if f["weight"] <= 0 or f["group"] in ("length_hist", "ai_tells", "phrases"):
            continue
        if name.startswith("opener:"):
            continue
        r = ref.get(name)
        if r is None:
            continue
        if name.startswith("fw:"):
            # function words are compared with general English (wordfreq), which is a
            # far larger baseline than the bundled reference text
            w = name[3:]
            base = wf_rate_per_1k(w)
            lr = math.log((f["value"] + 0.05) / (base + 0.05))
            if f["value"] < 0.5 and base < 0.5:
                continue
            # damp rare words: a big ratio on a word used 0.3 times per 1k is not a habit
            lr *= min(1.0, math.sqrt(max(f["value"], base) / 2.0))
            cands.append((abs(lr) * 1.4 * f["weight"], name, f["value"], base,
                          f"'{w}' {f['value']:.1f}/1k vs {base:.1f}/1k in general English"
                          f" ({'more' if lr > 0 else 'less'} often)"))
            continue
        sd = max(f["between_sd"], expected_noise_sd(f, f["value"], f["n"]), 1e-6,
                 0.05 * abs(f["value"]))
        z = (f["value"] - r["value"]) / sd
        z = max(-6, min(6, z))
        label = PLAIN.get(name, name)
        cands.append((abs(z) * f["weight"] * 0.5, name, f["value"], r["value"],
                      f"{label}: {fmt_val(name, f['value'])} (generic prose: {fmt_val(name, r['value'])})"))
    cands.sort(key=lambda x: -x[0])
    out, groups_seen = [], Counter()
    for score, name, v, rv, text in cands:
        g = feats[name]["group"]
        if groups_seen[g] >= (3 if g == "function_words" else 2):
            continue
        groups_seen[g] += 1
        out.append(text)
        if len(out) >= k:
            break
    return out


def summarize(prof: dict) -> str:
    ref = reference_features(prof["selection"])
    corp = prof["corpus"]
    m = prof["meta"]
    lines = [f"# Voice profile: {m.get('voice')}/{m.get('genre')}",
             f"Corpus: {corp['words']:,} words, {corp['sentences']:,} sentences, "
             f"{corp['paragraphs']:,} paragraphs, {corp['samples']} samples"
             + (" (pseudo-samples: the corpus was split to estimate variance)" if m.get("pseudo_samples") else "")
             + f", from {len(m.get('sources', []))} file(s)."]
    f = prof["features"]
    lines.append(f"Sentence length: mean {f['sent_len_mean']['value']:.1f}, SD {f['sent_len_sd']['value']:.1f}, "
                 f"median {f['sent_len_p50']['value']:.0f} (p10 {f['sent_len_p10']['value']:.0f}, "
                 f"p90 {f['sent_len_p90']['value']:.0f}); short {100 * f['sent_short_share']['value']:.0f}%, "
                 f"long {100 * f['sent_long_share']['value']:.0f}%.")
    lines.append("\n## Most distinctive habits (vs. generic/AI-style prose)")
    lines += [f"- {h}" for h in distinctive_habits(prof, ref)]
    ph = prof["selection"]["phrases"][:10]
    if ph:
        lines.append("\n## Signature phrases")
        lines.append(", ".join(f"\"{p['phrase']}\" ({p['count']})" for p in ph))
    op = list(prof["opener_distribution"].items())[:8]
    lines.append("\n## Common sentence openers")
    lines.append(", ".join(f"{k} {100 * v:.1f}%" for k, v in op))
    if prof["warnings"]:
        lines.append("\n## Warnings")
        lines += [f"- {w}" for w in prof["warnings"]]
    return "\n".join(lines)


# ======================================================================= main

def run_profile(paths, voice: str, genre: str = "general", write: bool = True,
                calibrate: bool = True) -> dict:
    files = ex.collect_files(paths)
    if not files:
        raise SystemExit("no supported files (.pdf .docx .md .txt) found in: " + " ".join(map(str, paths)))
    docs = []
    for fpath in files:
        d = ex.extract(fpath)
        ex.log_document(d)
        if d.n_words < 50:
            log(f"[profile] skipping {fpath.name}: too little text after cleaning")
            continue
        docs.append(d)
    if not docs:
        raise SystemExit("no usable text extracted")
    groups, names, cits, pseudo = make_samples(docs)
    log(f"[profile] parsing {sum(d.n_words for d in docs):,} words in {len(groups)} samples ...")
    para_counts = [paragraph_counts(g) for g in groups]
    sample_counts = [finalize_counts(pc, c) for pc, c in zip(para_counts, cits)]
    meta = {"voice": voice, "genre": genre,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "sources": [str(Path(d.path).resolve()) for d in docs],
            "samples": names, "pseudo_samples": pseudo,
            "extraction": [d.summary() for d in docs]}
    prof = build_profile(sample_counts, meta)
    if write:
        vd = voice_dir(voice, genre, create=True)
        write_json(vd / "profile.json", prof)
        write_json(vd / "samples.json", {"names": names, "counts": [counts_to_json(c) for c in sample_counts]})
        ex_recs = build_exemplars(groups, names, prof["selection"]["function_words"][:EXEMPLAR_FW])
        with open(vd / "exemplars.jsonl", "w", encoding="utf-8") as fh:
            for r in ex_recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (vd / "sources.txt").write_text("\n".join(meta["sources"]) + "\n", encoding="utf-8")
        log(f"[profile] wrote {vd} ({len(ex_recs)} exemplars)")
        if calibrate:
            import score
            cal = score.calibrate(para_counts, sample_counts, names, prof)
            write_json(vd / "calibration.json", cal)
            prof["_calibration"] = cal
            log(f"[profile] wrote calibration.json")
    return prof


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--genre", default="general")
    ap.add_argument("--json", action="store_true", help="print profile JSON instead of the summary")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--add", action="store_true",
                    help="add these documents to the voice's existing sources and rebuild")
    a = ap.parse_args(argv)
    paths = list(a.paths)
    if a.add:
        src = voice_dir(a.voice, a.genre) / "sources.txt"
        if src.exists():
            old = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
            gone = [p for p in old if not Path(p).exists()]
            if gone:
                log(f"[profile] {len(gone)} earlier source file(s) no longer exist and are dropped: {gone[:3]}")
            paths = [p for p in old if Path(p).exists()] + paths
    prof = run_profile(paths, a.voice, a.genre, write=not a.no_write, calibrate=not a.no_calibrate)
    if a.json:
        print(json.dumps(prof, indent=1))
    else:
        print(summarize(prof))
        if "_calibration" in prof:
            import score
            print("\n" + score.render_calibration(prof["_calibration"]))


if __name__ == "__main__":
    main()
