"""Score a text against a voice profile; calibrate the within-author band.

    python score.py --voice NAME [--genre G] FILE [--gaps 8] [--json]
    python score.py --voice NAME [--genre G] --calibrate        (re-runs calibration
                                                                from sources.txt)

Output: per-feature value / target / z / weight, Burrows' Delta, Cosine Delta, a
weighted composite distance, the length-matched within-author band from
calibration.json, and a gap report phrased as numeric instructions.

z-scores use  sd_eff = sqrt(between_sample_sd^2 + sampling_noise(text_length)^2):
a short text is *expected* to wander more, so it is not punished for noise, and
features that cannot be estimated at that length are reported as unmeasurable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, log, read_json, voice_dir, wf_rate_per_1k, write_json  # noqa: E402
import extract as ex  # noqa: E402
import tells  # noqa: E402
import voiceprofile as vp  # noqa: E402

Z_CAP = 6.0
SHORT_TEXT_WORDS = 300
MIN_EXPECTED_EVENTS = 2.0
CAL_LENGTHS = [100, 250, 500, 1000, 2000, 4000]
EXTRA_GROUP_WEIGHTS = {"length_hist": 0.5, "openers": 0.5, "paragraphs": 0.0}
# within-group down-weighting of noisy features
FEATURE_WEIGHT = {"sent_len_autocorr": 0.5}
# rhythm features allowed to lead the gap report
LEAD_RHYTHM = {"sent_len_w1", "sent_len_sd", "sent_loglen_sd", "sent_short_share", "sent_long_share"}
DELTA_TOP = 100

# Features that need many units: not estimable on short texts.
NEEDS_LENGTH = ("sent_len_p", "sent_len_autocorr", "para_len_", "mattr_500", "sent_hist_",
                "dep_depth_mean")


AGG = "mean"  # how |z| values are pooled within a group: "mean" (best AI-vs-author separation in tests) or "rms"


def _agg(v: list[tuple[float, float]]) -> float:
    sw = sum(w for _, w in v)
    if AGG == "rms":
        return math.sqrt(sum(w * z * z for z, w in v) / sw)
    return sum(w * z for z, w in v) / sw


def score_group(name: str, f: dict) -> str:
    if name.startswith("opener"):
        return "openers"
    if name.startswith("para_len_"):
        return "paragraphs"
    return f["group"]


# ======================================================================= scoring

def measurable(name: str, pf: dict, tf: dict, words: int) -> tuple[bool, str]:
    if words < SHORT_TEXT_WORDS and name.startswith(NEEDS_LENGTH):
        return False, f"needs >= {SHORT_TEXT_WORDS} words"
    k = pf["kind"]
    if k in ("rate", "prop"):
        n = tf["n"]
        if k == "rate":
            expected = pf["value"] * n / pf.get("scale", 1000.0)
        else:
            expected = pf["value"] * n
        need = MIN_EXPECTED_EVENTS
        if name.startswith("tell:") and name[5:] in tells.STRONG:
            need = 1.0      # strong tells justify action on one sighting (humanizer catalog)
        if max(expected, tf["count"]) < need:
            return False, "too rare to estimate at this length"
    if k in ("stat", "mean") and tf["n"] < 3:
        return False, "too few units"
    return True, ""


def feature_z(name: str, pf: dict, x: float, n_text: float) -> tuple[float, float]:
    target = pf["value"]
    noise = vp.expected_noise_sd(pf, target, n_text)
    if pf.get("log") and "between_sd_log" in pf and x > 0 and target > 0:
        if pf["kind"] == "stat":   # noise model for 'stat' scales the between-sample spread
            ns = float(np.mean(pf.get("per_sample_n") or [n_text]))
            noise_log = pf["between_sd_log"] * math.sqrt(max(1.0, ns / max(1.0, n_text)))
        else:
            noise_log = noise / target
        sd = math.sqrt(pf["between_sd_log"] ** 2 + noise_log ** 2)
        sd = max(sd, 0.02)
        return (math.log(x) - math.log(target)) / sd, sd
    sd = math.sqrt(pf.get("between_sd", 0.0) ** 2 + noise ** 2)
    sd = max(sd, 1e-3 + 0.02 * abs(target))
    z = (x - target) / sd
    if pf["group"] == "ai_tells":
        z = max(0.0, z)  # only an excess over the author counts
    return z, sd


def burrows_delta(prof: dict, text_feats: dict) -> tuple[float, int]:
    fw = [n for n, f in prof["features"].items() if n.startswith("fw:") and f["reliability"] != "low"]
    fw.sort(key=lambda n: -prof["features"][n]["value"])
    zs = []
    for n in fw[:DELTA_TOP]:
        f = prof["features"][n]
        vals = f["per_sample"]
        mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if sd <= 0:
            continue
        zs.append(abs((text_feats[n]["value"] - mu) / sd))
    return (float(np.mean(zs)) if zs else float("nan")), len(zs)


def cosine_delta(prof: dict, text_feats: dict) -> float:
    """1 - cos(z_text, z_author), both standardized against general English
    (wordfreq) with the author's between-sample SD."""
    fw = [n for n, f in prof["features"].items() if n.startswith("fw:") and f["reliability"] != "low"]
    fw.sort(key=lambda n: -prof["features"][n]["value"])
    a, b = [], []
    for n in fw[:DELTA_TOP]:
        f = prof["features"][n]
        vals = f["per_sample"]
        sd = max(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 0.05)  # classical SD, as in Delta
        ref = wf_rate_per_1k(n[3:])
        a.append((text_feats[n]["value"] - ref) / sd)
        b.append((f["value"] - ref) / sd)
    a, b = np.array(a), np.array(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1 - a @ b / den) if den > 0 else float("nan")


# ----------------------------------------------------------------- rhythm

W1_Q = np.linspace(0.01, 0.99, 99)
W1_REPS = 200
MIN_RHYTHM_SENTS = 6
RHYTHM_GAP_BOOST = 1.5
LEN_BINS = [(1, 9, "under 10 words"), (10, 19, "10-19 words"), (20, 34, "20-34 words"), (35, 10_000, "35+ words")]


def _logq(x) -> np.ndarray:
    """Quantiles of log length, centred on the median: the distance then measures the
    *shape* of the distribution (spread, tails) - mean length is its own feature."""
    q = np.quantile(np.log(np.asarray(x, dtype=float)), W1_Q)
    return q - np.median(np.log(np.asarray(x, dtype=float)))


def w1_log(a, ref_q: np.ndarray) -> float:
    """1-D Wasserstein distance between median-centred log sentence-length distributions."""
    return float(np.mean(np.abs(_logq(a) - ref_q)))


_NULL_CACHE: dict = {}


def rhythm_null(prof: dict, n: int) -> tuple[float, float]:
    """Mean/SD of W1 for an n-sentence passage of the author's own writing:
    contiguous runs of paragraphs from a randomly chosen sample (so between-document
    variation is included), compared with the pooled distribution."""
    key = (id(prof), int(round(n / 5.0)) * 5 if n > 40 else n)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    samples = [s for s in prof["sentence_lengths"] if s]
    pooled = [x for s in samples for p in s for x in p if x > 0]
    ref_q = _logq(pooled)
    rng = np.random.default_rng(12345)
    sizes = np.array([sum(len(p) for p in s) for s in samples], dtype=float)
    vals = []
    for _ in range(W1_REPS):
        got: list[int] = []
        while len(got) < n:
            si = rng.choice(len(samples), p=sizes / sizes.sum())
            paras = samples[si]
            start = int(rng.integers(len(paras)))
            for k in range(len(paras)):
                got.extend(x for x in paras[(start + k) % len(paras)] if x > 0)
                if len(got) >= n:
                    break
        vals.append(w1_log(got[:n], ref_q))
    res = (float(np.mean(vals)), float(np.std(vals)) or 1e-3)
    _NULL_CACHE[key] = res
    return res


def rhythm_row(counts: dict, prof: dict) -> dict | None:
    sl = [x for x in counts["sent_lens"] if x > 0]
    if len(sl) < MIN_RHYTHM_SENTS or not prof.get("sentence_lengths"):
        return None
    pooled = [x for s in prof["sentence_lengths"] for p in s for x in p if x > 0]
    w1 = w1_log(sl, _logq(pooled))
    mu, sd = rhythm_null(prof, len(sl))
    z = max(0.0, (w1 - mu) / sd)
    t = np.asarray(sl)
    a = np.asarray(pooled)
    bins = [{"label": lab, "text": float(((t >= lo) & (t <= hi)).mean()),
             "author": float(((a >= lo) & (a <= hi)).mean())} for lo, hi, lab in LEN_BINS]
    return {"value": w1, "target": mu, "z": z, "sd_eff": sd, "weight": 1.0, "group": "rhythm",
            "reliability": "high", "count": len(sl), "n": len(sl), "bins": bins}


def score_counts(counts: dict, prof: dict, n_gaps: int = 8, band: dict | None = None) -> dict:
    sel = prof["selection"]
    tf = vp.compute_features(counts, sel["function_words"], [p["phrase"] for p in sel["phrases"]],
                             sel["openers"])
    W = counts["words"]
    gw = dict(prof.get("group_weights", vp.GROUP_WEIGHTS), **EXTRA_GROUP_WEIGHTS)
    rows, unmeasurable = {}, []
    by_group: dict[str, list[tuple[float, float]]] = {}
    for name, pf in prof["features"].items():
        t = tf[name]
        weight = pf["weight"]
        if pf["group"] == "ai_tells":
            weight = 1.0       # always checked; target is the author's own (often ~0) rate
        weight *= FEATURE_WEIGHT.get(name, 1.0)
        if weight <= 0:
            continue
        ok, why = measurable(name, pf, t, W)
        if not ok:
            unmeasurable.append({"feature": name, "reason": why})
            continue
        z, sd = feature_z(name, pf, t["value"], t["n"])
        g = score_group(name, pf)
        rows[name] = {"value": t["value"], "target": pf["value"], "z": z, "sd_eff": sd,
                      "weight": weight, "group": g, "reliability": pf["reliability"],
                      "count": t["count"], "n": t["n"]}
        by_group.setdefault(g, []).append((min(abs(z), Z_CAP), weight))
    rr = rhythm_row(counts, prof)
    if rr is not None:
        rows["sent_len_w1"] = rr
        # the distribution distance counts as much as all other rhythm features together
        others = by_group.get("rhythm", [])
        by_group.setdefault("rhythm", []).append((min(rr["z"], Z_CAP), max(1.0, sum(w for _, w in others))))
    else:
        unmeasurable.append({"feature": "sent_len_w1", "reason": f"needs >= {MIN_RHYTHM_SENTS} sentences"})
    group_scores = {g: _agg(v) for g, v in by_group.items()}
    tot_w = sum(gw.get(g, 1.0) for g in group_scores)  # zero-weight groups are reported only
    composite = sum(gw.get(g, 1.0) * s for g, s in group_scores.items()) / tot_w if tot_w else float("nan")
    delta, n_delta = burrows_delta(prof, tf)
    out = {
        "words": W, "sentences": counts["sents"],
        "composite": composite, "burrows_delta": delta, "delta_words": n_delta,
        "cosine_delta": cosine_delta(prof, tf),
        "group_scores": group_scores, "features": rows,
        "unmeasurable": unmeasurable,
        "short_text": W < SHORT_TEXT_WORDS,
    }
    if band:
        b = band_for_length(band, W)
        out["band"] = b
        if b:
            out["inside_band"] = composite <= b["hi"]
            if b.get("rhythm_hi") is not None and "rhythm" in group_scores:
                out["rhythm_inside_band"] = group_scores["rhythm"] <= b["rhythm_hi"]
    out["gaps"] = gap_report(out, prof, n_gaps, force_rhythm=out.get("rhythm_inside_band") is False)
    return out


# ======================================================================= gaps

SUBSTITUTES = {
    "however": ["but", "yet", "still"], "moreover": ["and", "also", "besides"],
    "furthermore": ["and", "also"], "additionally": ["and", "also"], "therefore": ["so", "then"],
    "thus": ["so", "then"], "hence": ["so"], "consequently": ["so"], "although": ["though", "but"],
    "nevertheless": ["still", "yet", "but"], "nonetheless": ["still", "yet"],
    "ultimately": ["in the end", "finally"], "overall": [], "notably": [], "importantly": [],
    "but": ["however", "yet"], "so": ["thus", "therefore"], "also": ["moreover", "and"],
    "which": ["that"], "that": ["which"], "while": ["whereas", "although"],
}

GAP_TEXT = {
    "passive_share": ("Passive voice", "% of clauses", "convert some passive clauses to active voice (name the actor)",
                      "recast some active clauses in the passive where the actor is unimportant"),
    "nominalization_rate": ("Nominalizations", "/1k", "turn -tion/-ment/-ness nouns back into verbs and adjectives",
                            "use more abstract -tion/-ment nouns"),
    "first_person_sing_rate": ("I/me/my", "/1k", "remove some first-person singular",
                               "where the source already takes a stance, voice it in the first person "
                               "(never invent personal experiences or opinions)"),
    "first_person_plur_rate": ("We/us/our", "/1k", "use fewer we/us/our",
                               "use 'we' where the source's subject allows (never invent claims)"),
    "second_person_rate": ("You/your", "/1k", "address the reader directly less",
                           "address the reader as 'you' where it fits naturally"),
    "hedge_rate": ("Hedges", "/1k", "cut some hedges (may, might, suggests, appears)",
                   "add hedges where the source's claim strength allows (do not weaken claims)"),
    "booster_rate": ("Boosters", "/1k", "cut boosters (clearly, demonstrates, certainly)",
                     "use more emphatic verbs where the source's claim strength allows"),
    "contraction_rate": ("Contractions", "/1k", "expand contractions", "use contractions (don't, it's, can't)"),
    "subord_per_sent": ("Subordinate clauses", "/sentence", "split complex sentences; fewer which/because/when clauses",
                        "combine sentences with subordinate clauses (which, because, when, although)"),
    "dep_depth_mean": ("Syntactic depth", "", "simplify sentence structure", "nest more clauses within sentences"),
    "word_len_mean": ("Mean word length", " letters", "prefer shorter, plainer words", "use longer words"),
    "polysyllabic_share": ("3+ syllable words", "%", "prefer shorter, plainer words", "use more polysyllabic words"),
    "latinate_share": ("Latinate words", "%", "swap Latinate words for plain Germanic ones (use > utilize, show > demonstrate)",
                       "use more Latinate vocabulary"),
    "mattr_500": ("Vocabulary variety (MATTR)", "", "repeat key words rather than cycling synonyms",
                  "vary word choice more"),
    "commas_per_sent": ("Commas", "/sentence", "use fewer commas; fewer appositives and asides",
                        "use more commas: introductory phrases, asides, series"),
    "para_len_words_mean": ("Paragraph length", " words", "split long paragraphs", "merge short paragraphs"),
    "para_len_sents_mean": ("Paragraph length", " sentences", "split long paragraphs", "merge short paragraphs"),
    "para_len_words_sd": ("Variation in paragraph length (SD)", " words", "make paragraph lengths more even",
                          "vary paragraph length more: some one- or two-sentence paragraphs, some long ones"),
    "para_len_sents_sd": ("Variation in paragraph length (SD)", " sentences", "make paragraph lengths more even",
                          "vary paragraph length more: some one- or two-sentence paragraphs, some long ones"),
    "para_transition_open_share": ("Paragraphs opening with a transition", "%",
                                   "open more paragraphs with a topic sentence instead of a connective",
                                   "open more paragraphs with a connective (But, Yet, So, Still)"),
    "para_summary_close_share": ("Paragraphs ending in a summary sentence", "%",
                                 "stop ending paragraphs with a summarizing sentence; end on the last concrete point",
                                 "close more paragraphs with a brief summing-up sentence"),
    "sent_short_share": ("Short sentences (<10 words)", "%", "merge some short sentences", "add some sentences under 10 words"),
    "sent_long_share": ("Long sentences (>35 words)", "%", "split some long sentences", "build some sentences over 35 words"),
    "sent_len_autocorr": ("Long/short alternation (autocorrelation)", "",
                          "alternate long and short sentences more", "group sentences of similar length together more"),
}
PUNCT_NAMES = {"semicolon": "Semicolons", "colon": "Colons", "dash": "Dashes", "paren": "Parentheses",
               "question": "Questions", "exclam": "Exclamation marks", "quote": "Quotation marks"}


def _num(name, v):
    if name.endswith("_share"):
        return f"{100 * v:.0f}%"
    return f"{v:.1f}" if abs(v) >= 1 or v == 0 else f"{v:.2f}"


def phrase_gap(name: str, r: dict, prof: dict, W: int) -> str:
    v, t = r["value"], r["target"]
    delta = (t - v) * W / 1000
    fe = prof["features"]

    if name.startswith("fw:"):
        w = name[3:]
        if v > t:
            alts = [a for a in SUBSTITUTES.get(w, []) if f"fw:{a}" in fe and fe[f"fw:{a}"]["value"] > 0.3]
            alt = ""
            if alts:
                alt = "; the author prefers " + " and ".join(
                    f"'{a}' ({fe[f'fw:{a}']['value']:.1f}/1k)" for a in alts[:2])
            verb = "Remove most instances" if t < 0.4 * v else f"Cut about {abs(delta):.0f}"
            return f"'{w}' at {v:.1f}/1k; target {t:.1f}. {verb}{alt}."
        return f"'{w}' at {v:.1f}/1k; target {t:.1f}. Use it about {abs(delta):.0f} more times where it fits naturally."
    if name.startswith("phrase:"):
        ph = name[7:]
        if v < t:
            return (f"Signature phrase \"{ph}\" at {v:.2f}/1k; target {t:.2f}. Use it "
                    f"~{max(1, round(abs(delta)))} time(s) where it fits naturally.")
        return f"\"{ph}\" at {v:.2f}/1k; target {t:.2f}. Use it less."
    if name.startswith("punct_"):
        k = name[6:]
        lab = PUNCT_NAMES.get(k, k)
        how = ("Join some closely related sentences" if k == "semicolon" else "Add some") if v < t else \
              ("Replace most with periods or commas" if k in ("semicolon", "dash", "colon") else "Use fewer")
        return f"{lab} at {v:.1f}/1k; target {t:.1f}. {how} (about {abs(delta):.0f} in this text)."
    if name.startswith("tell:"):
        meta = tells.PATTERNS[name[5:]]
        return (f"AI tell {name[5:]} ({meta['name']}, {meta['tier']}): {v:.1f}/1k; the author's own rate is "
                f"{t:.1f}/1k. {meta['fix']}")
    if name == "sent_len_sd":
        p10 = fe["sent_len_p10"]["value"]
        p90 = fe["sent_len_p90"]["value"]
        if v < t:
            return (f"Sentence length SD is {v:.1f}; target {t:.1f}. Add some sentences under "
                    f"{max(4, round(p10) + 1)} words and some over {round(p90)}.")
        return f"Sentence length SD is {v:.1f}; target {t:.1f}. Make sentence lengths more even."
    if name == "sent_len_w1":
        diffs = sorted(r["bins"], key=lambda b: -abs(b["text"] - b["author"]))
        parts = [f"{b['label']}: {100 * b['text']:.0f}% vs the author's {100 * b['author']:.0f}%"
                 for b in diffs[:2] if abs(b["text"] - b["author"]) >= 0.04]
        fix = "; ".join(parts) if parts else "the overall shape differs"
        return (f"Sentence-length distribution is off (distance {v:.2f}; the author's own passages of this "
                f"length typically reach {t:.2f}). Biggest differences: {fix}. Re-balance sentence lengths.")
    if name == "sent_loglen_sd":
        return (f"Sentence-length spread (SD of log length) is {v:.2f}; target {t:.2f}. "
                + ("Mix in much shorter and much longer sentences." if v < t else "Make sentence lengths more even."))
    if name == "sent_len_mean":
        return (f"Mean sentence length is {v:.1f} words; target {t:.1f}. "
                + ("Combine some sentences." if v < t else "Split some long sentences."))
    if name.startswith("sent_len_p"):
        return f"{name[9:].upper()} sentence length is {v:.0f} words; target {t:.0f}."
    if name.startswith("opener_pos:"):
        pos = name.split(":")[1]
        return (f"Sentences opening with {vp.POS_PLAIN.get(pos, pos)}: {100 * v:.0f}%; target {100 * t:.0f}%. "
                + ("Open fewer sentences this way." if v > t else "Open more sentences this way."))
    if name.startswith("opener:"):
        o = name.split(":", 1)[1]
        return (f"Sentences opening with \"{o}\": {100 * v:.1f}%; target {100 * t:.1f}%. "
                + ("Use this opener less." if v > t else "Use this opener more."))
    if name.startswith("sent_hist_"):
        return f"Share of sentences {name[10:].replace('_', '-')} words: {100 * v:.0f}%; target {100 * t:.0f}%."
    if name in GAP_TEXT:
        lab, unit, down, up = GAP_TEXT[name]
        fmt = (lambda x: f"{100 * x:.0f}") if unit.startswith("%") else (lambda x: f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}")
        act = down if v > t else up
        return f"{lab} at {fmt(v)}{unit}; target {fmt(t)}{unit}. {act[0].upper() + act[1:]}."
    return f"{name} is {v:.2f}; target {t:.2f}."


# scored, but never turned into rewrite instructions: they depend on content, and
# "fixing" them would mean inventing quotations, lists or citations
# (paragraph length is fixed: rewrites keep the source's paragraph count)
# function words that carry meaning (modality, negation, quantity): changing their rate
# would change claim strength, so they are scored but never turned into instructions
MEANING_WORDS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would", "not", "no",
                 "never", "nor", "all", "some", "any", "every", "each", "most", "many", "few", "more",
                 "less", "only", "always", "often", "sometimes", "usually"}
NOT_ACTIONABLE = {"punct_quote", "list_item_rate", "para_len_words_mean", "para_len_sents_mean",
                  "para_len_words_sd", "para_len_sents_sd"}


def gap_report(sc: dict, prof: dict, n: int, force_rhythm: bool = False) -> list[dict]:
    cands = []
    for name, r in sc["features"].items():
        if r["group"] in ("length_hist", "paragraphs") or name in NOT_ACTIONABLE or name.startswith("citation"):
            continue
        if name.startswith("fw:") and name[3:] in MEANING_WORDS:
            continue
        pri = min(abs(r["z"]), Z_CAP * 1.5) * r["weight"]
        if r["group"] == "rhythm" and abs(r["z"]) >= 1.0:
            pri *= RHYTHM_GAP_BOOST
        if pri < 1.0 and not (force_rhythm and name in LEAD_RHYTHM and abs(r["z"]) >= 0.75):
            continue
        cands.append((pri, name, r))
    cands.sort(key=lambda x: -x[0])
    # sentence rhythm is the most visible voice trait: when it is clearly off (|z| >= 2),
    # its strongest gap always leads the report
    # (also whenever the rhythm group is outside its calibrated band)
    rhythm = [c for c in cands if c[1] in LEAD_RHYTHM and (abs(c[2]["z"]) >= 2.0 or force_rhythm)]
    rhythm.sort(key=lambda c: -abs(c[2]["z"]))
    if rhythm:
        cands.remove(rhythm[0])
        cands.insert(0, rhythm[0])
    out, seen_len = [], 0
    for pri, name, r in cands:
        if name.startswith("sent_len_p"):
            if seen_len >= 1:   # percentiles are redundant with SD / mean
                continue
            seen_len += 1
        out.append({"feature": name, "z": round(r["z"], 2), "priority": round(pri, 2),
                    "value": r["value"], "target": r["target"],
                    "instruction": phrase_gap(name, r, prof, sc["words"])})
        if len(out) >= n:
            break
    return out


# ======================================================================= calibration

def chunk_paragraph_counts(pcs: list[dict], L: int) -> list[list[dict]]:
    chunks, cur, w = [], [], 0
    for pc in pcs:
        if pc["words"] == 0:
            continue
        cur.append(pc)
        w += pc["words"]
        if w >= L:
            chunks.append(cur)
            cur, w = [], 0
    if cur and w >= 0.7 * L:
        chunks.append(cur)
    return chunks


def _band_stats(xs: list[float]) -> dict:
    a = np.array([x for x in xs if not math.isnan(x)])
    if not len(a):
        return {}
    return {"n": int(len(a)), "lo": float(np.percentile(a, 10)), "median": float(np.median(a)),
            "hi": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def calibrate(para_counts: list[list[dict]], sample_counts: list[dict], names: list[str],
              prof: dict, max_chunks: int = 150) -> dict:
    """Leave-one-out: score each sample (whole, and in length-matched chunks)
    against a profile built from the other samples."""
    k = len(sample_counts)
    per_len: dict[str, dict[str, list]] = {}
    loo_full = []
    if k < 2:
        return {"error": "need >= 2 samples for calibration"}
    for i in range(k):
        others = [c for j, c in enumerate(sample_counts) if j != i]
        p_i = vp.build_profile(others)
        full = score_counts(sample_counts[i], p_i, n_gaps=0)
        loo_full.append({"sample": names[i], "words": sample_counts[i]["words"],
                         "composite": full["composite"], "burrows_delta": full["burrows_delta"],
                         "cosine_delta": full["cosine_delta"],
                         "rhythm": full["group_scores"].get("rhythm", float("nan"))})
        for L in CAL_LENGTHS:
            chunks = chunk_paragraph_counts(para_counts[i], L)
            step = max(1, math.ceil(len(chunks) * k / max_chunks))
            for ch in chunks[::step]:
                s = score_counts(vp.finalize_counts(ch, prune=False), p_i, n_gaps=0)
                d = per_len.setdefault(str(L), {"composite": [], "burrows_delta": [], "cosine_delta": [],
                                                "rhythm": [], "words": []})
                for m in ("composite", "burrows_delta", "cosine_delta"):
                    d[m].append(s[m])
                d["rhythm"].append(s["group_scores"].get("rhythm", float("nan")))
                d["words"].append(s["words"])
    bands = {}
    for L, d in per_len.items():
        bands[L] = {m: _band_stats(d[m]) for m in ("composite", "burrows_delta", "cosine_delta", "rhythm")}
        bands[L]["mean_words"] = float(np.mean(d["words"]))
    full_words = float(np.mean([r["words"] for r in loo_full]))
    bands["full"] = {m: _band_stats([r[m] for r in loo_full])
                     for m in ("composite", "burrows_delta", "cosine_delta", "rhythm")}
    bands["full"]["mean_words"] = full_words
    cal = {"method": "leave-one-out over samples; chunks never split paragraphs",
           "samples": k, "loo": loo_full, "bands": bands}
    cal["baseline"] = baseline_scores(prof)
    return cal


def baseline_scores(prof: dict) -> dict:
    """Distance of the bundled generic/AI-style text from the profile."""
    ref = DATA_DIR / "reference_generic.md"
    if not ref.exists():
        return {}
    doc = ex.extract(ref)
    pcs = vp.paragraph_counts(doc.paragraphs)
    out = {"source": ref.name}
    whole = score_counts(vp.finalize_counts(pcs, prune=False), prof, n_gaps=0)
    out["full"] = {"words": whole["words"], "composite": whole["composite"],
                   "burrows_delta": whole["burrows_delta"], "cosine_delta": whole["cosine_delta"],
                   "rhythm": whole["group_scores"].get("rhythm", float("nan"))}
    for L in (100, 250, 500, 1000):
        vals = [score_counts(vp.finalize_counts(ch, prune=False), prof, n_gaps=0)
                for ch in chunk_paragraph_counts(pcs, L)]
        if vals:
            out[str(L)] = {m: float(np.median([v[m] for v in vals])) for m in
                           ("composite", "burrows_delta", "cosine_delta")}
            out[str(L)]["rhythm"] = float(np.nanmedian([v["group_scores"].get("rhythm", float("nan")) for v in vals]))
            out[str(L)]["n"] = len(vals)
    return out


def band_for_length(cal_or_bands: dict, words: int) -> dict | None:
    bands = cal_or_bands.get("bands", cal_or_bands)
    best, best_d = None, None
    for key, b in bands.items():
        mw = b.get("mean_words")
        if not mw or not b.get("composite"):
            continue
        d = abs(math.log(max(words, 50) / mw))
        if best_d is None or d < best_d:
            best, best_d = key, d
    if best is None:
        return None
    c = bands[best]["composite"]
    return {"length_bucket": best, "bucket_words": round(bands[best]["mean_words"]),
            "lo": c["lo"], "median": c["median"], "hi": c["hi"], "p95": c["p95"],
            "delta_hi": bands[best]["burrows_delta"].get("hi"),
            "rhythm_hi": (bands[best].get("rhythm") or {}).get("hi")}


# ======================================================================= I/O

def load_voice(voice: str, genre: str = "general") -> tuple[dict, dict | None]:
    vd = voice_dir(voice, genre)
    if not (vd / "profile.json").exists():
        raise SystemExit(f"no profile at {vd}; run voiceprofile.py first")
    cal = read_json(vd / "calibration.json") if (vd / "calibration.json").exists() else None
    return read_json(vd / "profile.json"), cal


def score_file(path, voice, genre="general", n_gaps=8) -> dict:
    prof, cal = load_voice(voice, genre)
    doc = ex.extract(path)
    counts = vp.finalize_counts(vp.paragraph_counts(doc.paragraphs), doc.citations, prune=False)
    return score_counts(counts, prof, n_gaps, cal)


def score_text(text: str, voice, genre="general", n_gaps=8, fmt="md") -> dict:
    prof, cal = load_voice(voice, genre)
    doc = ex.extract_text(text, fmt=fmt)
    counts = vp.finalize_counts(vp.paragraph_counts(doc.paragraphs), doc.citations, prune=False)
    return score_counts(counts, prof, n_gaps, cal)


def render(sc: dict, title: str = "") -> str:
    L = [f"# Style score{': ' + title if title else ''}",
         f"Words: {sc['words']:,}   composite distance: {sc['composite']:.2f}   "
         f"Burrows' Delta: {sc['burrows_delta']:.2f}   Cosine Delta: {sc['cosine_delta']:.2f}"]
    b = sc.get("band")
    if b:
        where = "INSIDE" if sc["composite"] <= b["hi"] else "OUTSIDE"
        L.append(f"Within-author band (length ~{b['bucket_words']} words): {b['lo']:.2f}-{b['hi']:.2f} "
                 f"(median {b['median']:.2f}) -> {where}")
        if sc["composite"] < b["lo"]:
            L.append("Note: below the band's low end. Closer than the author's own texts suggests over-fitting.")
    else:
        L.append("No calibration.json for this voice; band unavailable.")
    r = sc["features"].get("sent_len_w1")
    if r is not None:
        rs = sc["group_scores"].get("rhythm", float("nan"))
        verdict = ""
        if b and b.get("rhythm_hi") is not None:
            verdict = f" -> {'INSIDE' if rs <= b['rhythm_hi'] else 'OUTSIDE'} (author band <= {b['rhythm_hi']:.2f})"
        bins = ", ".join(f"{x['label']} {100 * x['text']:.0f}% (author {100 * x['author']:.0f}%)" for x in r["bins"])
        L.append(f"Sentence rhythm: {rs:.2f}{verdict}\n  {bins}")
    if sc["short_text"]:
        L.append(f"Short text (<{SHORT_TEXT_WORDS} words): only features estimable at this length were scored.")
    if sc["unmeasurable"]:
        L.append(f"Not measurable at this length: {len(sc['unmeasurable'])} features "
                 f"(e.g. {', '.join(u['feature'] for u in sc['unmeasurable'][:6])}).")
    L.append("\nGroup distances: " + ", ".join(f"{g} {v:.2f}" for g, v in
                                                 sorted(sc["group_scores"].items(), key=lambda x: -x[1])))
    if sc["gaps"]:
        L.append("\n## Top gaps")
        L += [f"{i + 1}. {g['instruction']}  (z={g['z']:+.1f})" for i, g in enumerate(sc["gaps"])]
    return "\n".join(L)


def recalibrate(voice: str, genre: str) -> dict:
    vd = voice_dir(voice, genre)
    srcs = [l for l in (vd / "sources.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [ex.extract(s) for s in srcs if Path(s).exists()]
    groups, names, cits, _ = vp.make_samples(docs)
    pcs = [vp.paragraph_counts(g) for g in groups]
    scs = [vp.finalize_counts(p, c) for p, c in zip(pcs, cits)]
    prof = read_json(vd / "profile.json")
    cal = calibrate(pcs, scs, names, prof)
    write_json(vd / "calibration.json", cal)
    return cal


def render_calibration(cal: dict) -> str:
    L = ["## Calibration (leave-one-out, composite distance)",
         "| length | n | p10 | median | p90 | max | Delta p90 | rhythm p90 |", "|---|---|---|---|---|---|---|---|"]
    for key, b in cal["bands"].items():
        c = b["composite"]
        if not c:
            continue
        L.append(f"| {key} (~{b['mean_words']:.0f} w) | {c['n']} | {c['lo']:.2f} | {c['median']:.2f} | "
                 f"{c['hi']:.2f} | {c['max']:.2f} | {b['burrows_delta'].get('hi', float('nan')):.2f} | "
                 f"{(b.get('rhythm') or {}).get('hi', float('nan')):.2f} |")
    base = cal.get("baseline") or {}
    if base:
        L.append("\nGeneric/AI-style reference text against this profile:")
        for key in ("100", "250", "500", "1000", "full"):
            if key in base:
                v = base[key]
                L.append(f"- {key}: composite {v['composite']:.2f}, Delta {v['burrows_delta']:.2f}, "
                         f"Cosine Delta {v['cosine_delta']:.2f}, rhythm {v.get('rhythm', float('nan')):.2f}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--genre", default="general")
    ap.add_argument("--gaps", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="re-run calibration from sources.txt")
    a = ap.parse_args(argv)
    if a.calibrate:
        cal = recalibrate(a.voice, a.genre)
        print(json.dumps(cal, indent=1) if a.json else render_calibration(cal))
        return
    if not a.file:
        ap.error("FILE is required unless --calibrate")
    sc = score_file(a.file, a.voice, a.genre, a.gaps)
    print(json.dumps(sc, indent=1) if a.json else render(sc, Path(a.file).name))


if __name__ == "__main__":
    main()
