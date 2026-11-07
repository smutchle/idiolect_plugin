"""Turn a numeric profile into a plain-English style brief (8-12 instructions).

    python brief.py --voice NAME [--genre G]

The brief is what goes into rewrite prompts. It never contains raw JSON: every
item is an instruction with the author's number and, where useful, the generic /
AI-prose number it departs from. Sentence rhythm always comes first.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, voice_dir, wf_rate_per_1k  # noqa: E402
import tells  # noqa: E402
import voiceprofile as vp  # noqa: E402

MAX_ITEMS = 12
LEN_BINS = [(1, 9, "under 10 words"), (10, 19, "10-19"), (20, 34, "20-34"), (35, 10_000, "35+")]
CONNECTIVES_TO_CHECK = ["however", "moreover", "furthermore", "additionally", "therefore", "thus",
                        "hence", "consequently", "nevertheless", "ultimately", "overall", "notably",
                        "importantly", "indeed", "also", "but", "yet", "so", "still", "though"]


def _f(prof, name):
    return prof["features"].get(name)


def _ok(f):
    return f is not None and f["weight"] > 0


def length_bins(prof) -> list[tuple[str, float]]:
    pooled = [x for s in prof.get("sentence_lengths", []) for p in s for x in p if x > 0]
    if not pooled:
        return []
    a = np.asarray(pooled)
    return [(lab, float(((a >= lo) & (a <= hi)).mean())) for lo, hi, lab in LEN_BINS]


def rhythm_item(prof, ref) -> str:
    f = prof["features"]
    p50, sd = f["sent_len_p50"]["value"], f["sent_len_sd"]["value"]
    p10, p90 = f["sent_len_p10"]["value"], f["sent_len_p90"]["value"]
    bins = length_bins(prof)
    bin_txt = ", ".join(f"{100 * v:.0f}% {lab}" for lab, v in bins)
    ref_sd = ref.get("sent_len_sd", {}).get("value")
    contrast = ""
    if ref_sd and sd > 1.4 * ref_sd:
        contrast = (f" Generic/AI prose is far more uniform (SD about {ref_sd:.0f}); do NOT even out sentence "
                    f"lengths. Write genuinely short sentences (under {max(5, round(p10) + 1)} words) and genuinely "
                    f"long ones (over {round(p90)}).")
    elif ref_sd and sd < 0.75 * ref_sd:
        contrast = " Keep lengths fairly even; avoid very short punchy sentences and very long ones."
    ac = f["sent_len_autocorr"]["value"]
    alt = ""
    if ac <= -0.05:
        alt = " Alternate long and short sentences."
    elif ac >= 0.15:
        alt = " Sentences of similar length tend to come in runs."
    return (f"**Sentence rhythm (most important).** Median sentence {p50:.0f} words, SD {sd:.0f}; "
            f"10% are {p10:.0f} words or fewer and 10% are {p90:.0f}+. Mix: {bin_txt}.{contrast}{alt}")


def paragraph_item(prof, ref):
    """Paragraph *count* is fixed by the source (fidelity alignment), so only the
    actionable paragraph habits are briefed: how paragraphs open and close."""
    f = prof["features"]
    extras = []
    sc = f.get("para_summary_close_share")
    r_sc = ref.get("para_summary_close_share", {}).get("value", 0)
    if sc is not None and sc["value"] < 0.08:
        extras.append(f"almost never ends a paragraph with a summing-up sentence ({100 * sc['value']:.0f}%); "
                      "end on the last concrete point")
    elif sc is not None and sc["value"] < 0.5 * max(r_sc, 0.01):
        extras.append("rarely closes paragraphs with a summary sentence")
    to = f.get("para_transition_open_share")
    r_to = ref.get("para_transition_open_share", {}).get("value")
    if _ok(to) and r_to is not None and abs(to["value"] - r_to) > 0.12:
        extras.append(f"opens {100 * to['value']:.0f}% of paragraphs with a connective (generic: {100 * r_to:.0f}%)")
    if not extras:
        return None
    return 1.0 + 0.5 * len(extras), "**Paragraph shape:** the author " + "; ".join(extras) + "."


PUNCT = {"punct_semicolon": ("Semicolons", "join closely related clauses with them"),
         "punct_colon": ("Colons", None), "punct_dash": ("Dashes", None),
         "punct_paren": ("Parentheses", None), "punct_question": ("Questions", "real questions, not set-ups"),
         "punct_exclam": ("Exclamation marks", None)}
# quotation marks are omitted on purpose: they mostly reflect quoted content, and
# asking for more of them invites invented quotations


def punctuation_items(prof, ref):
    out = []
    f = prof["features"]
    for name, (lab, how) in PUNCT.items():
        a = f.get(name)
        if a is None:
            continue
        r = ref.get(name, {}).get("value", 0.0)
        v = a["value"]
        lr = math.log((v + 0.3) / (r + 0.3))
        if abs(lr) < 0.6:
            continue
        sents_per = (f["sent_len_mean"]["value"] * 1000 / (v * f["sent_len_mean"]["value"])) if v > 0 else None
        if v > r:
            if not _ok(a):
                continue
            every = f" (about one every {1000 / v / f['sent_len_mean']['value']:.0f} sentences)" if v > 0.5 else ""
            txt = f"**{lab}:** {v:.1f} per 1,000 words{every}; generic prose {r:.1f}." + (f" Use them: {how}." if how else "")
        else:
            txt = f"**{lab}:** rare: {v:.1f} per 1,000 words (generic prose {r:.1f}). Avoid them."
        del sents_per
        out.append((abs(lr) * 0.9, txt))
    c = f["commas_per_sent"]["value"]
    rc = ref.get("commas_per_sent", {}).get("value", c)
    if abs(math.log((c + 0.1) / (rc + 0.1))) > 0.3:
        out.append((abs(math.log((c + 0.1) / (rc + 0.1))),
                    f"**Commas:** {c:.1f} per sentence (generic prose {rc:.1f})."))
    return out


def function_word_items(prof):
    f = prof["features"]
    favored, avoided = [], []
    for name, a in f.items():
        if not name.startswith("fw:") or a["weight"] <= 0:
            continue
        w = name[3:]
        g = wf_rate_per_1k(w)
        v = a["value"]
        if v >= 1.0 and g > 0.05:
            lr = math.log((v + 0.1) / (g + 0.1))
            if lr > 0.7:
                favored.append((lr * min(1, math.sqrt(v / 2)), w, v, g))
    for w in CONNECTIVES_TO_CHECK:
        a = f.get(f"fw:{w}")
        g = wf_rate_per_1k(w)
        if a is None or g < 0.1:
            continue
        v = a["value"]
        if v < 0.4 * g:
            avoided.append((math.log((g + 0.05) / (v + 0.05)), w, v, g))
    favored.sort(reverse=True)
    avoided.sort(reverse=True)
    items = []
    if favored:
        items.append((2.0 + favored[0][0] * 0.2, "**Favored words:** " + ", ".join(
            f"'{w}' ({v:.1f}/1k vs {g:.1f} in general English)" for _, w, v, g in favored[:6])
            + ". Let these carry the connective and logical work."))
    if avoided:
        items.append((2.0 + avoided[0][0] * 0.2, "**Avoided words:** " + ", ".join(
            f"'{w}' ({v:.2f}/1k vs {g:.1f} typical)" for _, w, v, g in avoided[:6])
            + ". Replace them with the author's own connectives."))
    return items


STANCE = [
    ("first_person_sing_rate", "I/me/my", "per 1,000 words"),
    ("first_person_plur_rate", "we/us/our", "per 1,000 words"),
    ("second_person_rate", "you/your", "per 1,000 words"),
    ("contraction_rate", "Contractions", "per 1,000 words"),
    ("hedge_rate", "Hedges (may, might, suggests)", "per 1,000 words"),
    ("booster_rate", "Boosters (clearly, demonstrates)", "per 1,000 words"),
]


def stance_items(prof, ref):
    out = []
    f = prof["features"]
    for name, lab, unit in STANCE:
        a = f.get(name)
        if a is None:
            continue
        r = ref.get(name, {}).get("value", 0.0)
        v = a["value"]
        lr = math.log((v + 0.5) / (r + 0.5))
        if abs(lr) < 0.5:
            continue
        if name == "contraction_rate" and v < 0.5:
            out.append((abs(lr), "**Contractions:** essentially never (write 'do not', 'it is')."))
            continue
        caveat = ""
        if name in ("first_person_sing_rate", "first_person_plur_rate", "second_person_rate"):
            caveat = (" Use this perspective only where the source already takes a stance; never invent "
                      "personal experiences, opinions or claims to hit the number.")
        out.append((abs(lr) * 0.8, f"**{lab}:** {v:.1f} {unit} (generic prose {r:.1f}).{caveat}"))
    p = f.get("passive_share")
    if p is not None:
        r = ref.get("passive_share", {}).get("value", p["value"])
        if abs(p["value"] - r) > 0.05:
            out.append((abs(p["value"] - r) * 8, f"**Passive voice:** {100 * p['value']:.0f}% of clauses "
                                                  f"(generic prose {100 * r:.0f}%)."))
    return out


def lexical_item(prof, ref):
    f = prof["features"]
    wl, rwl = f["word_len_mean"]["value"], ref.get("word_len_mean", {}).get("value")
    lat, rlat = f["latinate_share"]["value"], ref.get("latinate_share", {}).get("value")
    nom, rnom = f["nominalization_rate"]["value"], ref.get("nominalization_rate", {}).get("value")
    if rwl is None:
        return None
    if wl < rwl - 0.25:
        return (abs(wl - rwl) * 3, f"**Plain vocabulary:** mean word length {wl:.1f} letters (generic prose {rwl:.1f}); "
                                    f"Latinate-suffix words {100 * lat:.0f}% (generic {100 * rlat:.0f}); "
                                    f"nominalizations {nom:.0f}/1k (generic {rnom:.0f}). Prefer short Germanic "
                                    f"words and verbs over -tion/-ment nouns.")
    if wl > rwl + 0.25:
        return (abs(wl - rwl) * 3, f"**Elevated vocabulary:** mean word length {wl:.1f} letters (generic {rwl:.1f}); "
                                    f"nominalizations {nom:.0f}/1k.")
    return None


def opener_item(prof):
    od = [(k, v) for k, v in prof["opener_distribution"].items() if not k.startswith("<")]
    if not od:
        return None
    top = ", ".join(f"\"{k.rstrip(',').capitalize()}{',' if k.endswith(',') else ''}\" {100 * v:.0f}%"
                    for k, v in od[:6])
    f = prof["features"]
    adv = f.get("opener_pos:ADV", {}).get("value", 0)
    extra = f" Only {100 * adv:.0f}% of sentences open with an adverb; avoid 'Moreover,'/'Additionally,' openers." \
        if adv < 0.08 else ""
    return (1.2, f"**Sentence openers:** most common are {top}.{extra}")


def phrase_item(prof):
    ph = prof["selection"].get("phrases", [])[:8]
    if not ph:
        return None
    return (1.0, "**Signature phrases** (use sparingly, only where natural, at most about once each per 1,000 "
                 "words): " + ", ".join(f"\"{p['phrase']}\"" for p in ph) + ".")


def avoid_item(prof):
    """Humanizer tells the author does not use -> avoid; tells the author *does* use at a
    real rate -> keep at the author's rate (a writing sample overrides the patterns)."""
    f = prof["features"]
    avoid, keep = [], []
    for pid, meta in tells.PATTERNS.items():
        a = f.get(f"tell:{pid}")
        if a is None:
            continue
        rate = a["value"]
        if a["count"] >= 10 and rate >= 0.5:
            keep.append(f"{meta['name'].lower()} ({rate:.1f}/1k)")
        elif rate <= 0.3:
            avoid.append(f"{pid} {meta['name'].lower()}")
    txt = ""
    if avoid:
        txt += ("**Do not use** these AI tells (the author does not; details in humanizer-patterns.md): "
                + "; ".join(avoid) + ".")
    if keep:
        txt += (" **Author habits that look like tells - keep them at the author's rate:** "
                + "; ".join(keep) + ".")
    return (1.5, txt.strip()) if txt else None


def style_brief(prof: dict) -> str:
    ref = vp.reference_features(prof["selection"])
    ref = {k: v for k, v in ref.items()}
    items: list[tuple[float, str]] = []
    pi = paragraph_item(prof, ref)
    if pi:
        items.append(pi)
    items += punctuation_items(prof, ref)
    items += function_word_items(prof)
    items += stance_items(prof, ref)
    for it in (lexical_item(prof, ref), opener_item(prof), phrase_item(prof), avoid_item(prof)):
        if it:
            items.append(it)
    items.sort(key=lambda x: -x[0])
    chosen = [rhythm_item(prof, ref)] + [t for _, t in items[:MAX_ITEMS - 1]]
    m = prof["meta"]
    head = (f"# Style brief: {m.get('voice')}/{m.get('genre')}\n"
            f"Built from {prof['corpus']['words']:,} words. Numbers are targets for the whole document, "
            f"not for every sentence.\n")
    return head + "\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chosen)) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--genre", default="general")
    a = ap.parse_args(argv)
    prof = read_json(voice_dir(a.voice, a.genre) / "profile.json")
    print(style_brief(prof))


if __name__ == "__main__":
    main()
