"""Retrieve exemplar paragraphs that show how the author writes *this kind* of passage.

    python retrieve.py --voice NAME [--genre G] [--role R] [--k 4] FILE
    python retrieve.py --voice NAME --text "..."            (or text on stdin with -)

Matching (never on topic words):
  1. rhetorical role (intro / methods / results / discussion / other): same-role
     candidates are preferred whenever enough exist;
  2. paragraph function (argument / description / method / transition);
  3. structure: TF-IDF over *masked* token sequences (function words kept, every
     content word replaced by its POS tag), a standardized vector of function-word and
     punctuation features, and similar paragraph length;
  4. rhythm typicality: paragraphs whose sentence-length spread is typical of the
     author are preferred (source sentence lengths are deliberately not matched).
Diversity: Maximal Marginal Relevance, so the 3-4 exemplars are not near-duplicates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_nlp, read_json, voice_dir  # noqa: E402
import extract as ex  # noqa: E402
import voiceprofile as vp  # noqa: E402

W_ROLE, W_FUNC, W_TFIDF, W_STRUCT, W_LEN, W_RHYTHM = 0.30, 0.15, 0.25, 0.10, 0.10, 0.10
MMR_LAMBDA = 0.7
POOL_PER_K = 6          # same-role filtering only when >= k*POOL_PER_K candidates exist
PUNCT_KEYS = ("semicolon", "colon", "comma", "question", "paren", "dash")


class Bank:
    def __init__(self, recs: list[dict], fw_top: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not recs:
            raise SystemExit("exemplar bank is empty; re-run voiceprofile.py with more text")
        self.recs = recs
        self.fw_top = fw_top
        self.vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=False,
                                   ngram_range=(1, 3), sublinear_tf=True, min_df=2)
        self.X = self.vec.fit_transform([r["masked"] for r in recs])
        S = np.array([self.struct_vec(r) for r in recs])
        self.mu = S.mean(axis=0)
        self.sd = S.std(axis=0) + 1e-6
        self.S = (S - self.mu) / self.sd
        # rhythm typicality: how close each paragraph's sentence-length spread (CV) is
        # to the author's median paragraph -> exemplars that model the author's rhythm
        cv = np.array([r["sent_len_sd"] / max(r["sent_len_mean"], 1.0) for r in recs if r["n_sents"] >= 3])
        med = float(np.median(cv)) if len(cv) else 0.5
        self.rhythm = np.array([
            math.exp(-abs(math.log(max(r["sent_len_sd"] / max(r["sent_len_mean"], 1.0), 0.05) / max(med, 0.05))))
            if r["n_sents"] >= 3 else 0.5 for r in recs])

    @staticmethod
    def struct_vec(r: dict) -> np.ndarray:
        # sentence lengths are deliberately NOT matched to the source: an AI-drafted
        # source has uniform sentences, and matching on that would pick the author's
        # least characteristic paragraphs (see rhythm typicality below instead)
        return np.array([math.log(max(r["n_words"], 1)),
                         *[r["punct"].get(k, 0.0) for k in PUNCT_KEYS], *r["fw"]], dtype=float)

    def query(self, q: dict, k: int = 4, exclude_ids=()) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        n = len(self.recs)
        idx = [i for i in range(n) if self.recs[i]["id"] not in set(exclude_ids)]
        qx = self.vec.transform([q["masked"]])
        s_tfidf = cosine_similarity(qx, self.X).ravel()
        qs = (self.struct_vec(q) - self.mu) / self.sd
        d = np.sqrt(((self.S - qs) ** 2).mean(axis=1))
        s_struct = 1.0 / (1.0 + d)
        s_len = np.exp(-np.abs(np.log(max(q["n_words"], 1) / np.array([max(r["n_words"], 1) for r in self.recs]))))
        role = q.get("role") or "other"
        s_role = np.array([1.0 if r["role"] == role else (0.4 if "other" in (r["role"], role) else 0.0)
                           for r in self.recs])
        s_func = np.array([1.0 if r["function"] == q["function"] else 0.0 for r in self.recs])
        rel = (W_ROLE * s_role + W_FUNC * s_func + W_TFIDF * s_tfidf + W_STRUCT * s_struct
               + W_LEN * s_len + W_RHYTHM * self.rhythm)

        same_role = [i for i in idx if self.recs[i]["role"] == role]
        pool = same_role if len(same_role) >= k * POOL_PER_K else idx
        # MMR over the 60 most relevant candidates
        pool = sorted(pool, key=lambda i: -rel[i])[:60]
        if not pool:
            return []
        sim = cosine_similarity(self.X[pool])
        chosen: list[int] = []
        while len(chosen) < min(k, len(pool)):
            best, best_v = None, -1e9
            for pi, i in enumerate(pool):
                if pi in chosen:
                    continue
                red = max((sim[pi, c] for c in chosen), default=0.0)
                v = MMR_LAMBDA * rel[i] - (1 - MMR_LAMBDA) * red
                if v > best_v:
                    best, best_v = pi, v
            chosen.append(best)
        out = []
        for pi in chosen:
            i = pool[pi]
            r = self.recs[i]
            out.append({"id": r["id"], "source": r["source"], "role": r["role"], "function": r["function"],
                        "n_words": r["n_words"], "relevance": round(float(rel[i]), 3),
                        "match": {"role": float(s_role[i]), "function": float(s_func[i]),
                                  "structure_tfidf": round(float(s_tfidf[i]), 3),
                                  "structure_features": round(float(s_struct[i]), 3),
                                  "length": round(float(s_len[i]), 3),
                                  "rhythm_typicality": round(float(self.rhythm[i]), 3)},
                        "text": r["text"]})
        return out


@lru_cache(maxsize=8)
def load_bank(voice: str, genre: str = "general") -> Bank:
    vd = voice_dir(voice, genre)
    if not (vd / "exemplars.jsonl").exists():
        raise SystemExit(f"no exemplar bank at {vd}; run voiceprofile.py first")
    recs = [json.loads(l) for l in (vd / "exemplars.jsonl").open(encoding="utf-8") if l.strip()]
    prof = read_json(vd / "profile.json")
    return Bank(recs, prof["selection"]["function_words"][:vp.EXEMPLAR_FW])


def describe_query(text: str, fw_top: list[str], role: str | None = None) -> dict:
    """Structural description of a source chunk (one or more paragraphs)."""
    doc = ex.extract_text(text)
    paras = doc.paragraphs
    joined = "\n\n".join(b.text for b in paras) or text
    if role is None:
        # a lone chunk has no document position, so only headings and explicit cues
        # count; the rewrite loop passes --role from the whole-document extraction
        heads = [b for b in doc.blocks if b.kind == "heading"]
        role = (ex.role_from_heading(heads[-1].text) if heads else None) or ex.cue_role(joined) or "other"
    sdoc = get_nlp()(joined)
    q = vp.exemplar_features(sdoc, joined, fw_top)
    # length similarity is per paragraph: use the mean paragraph length of the chunk
    if len(paras) > 1:
        q["n_words"] = int(q["n_words"] / len(paras))
    q["role"] = role
    return q


def retrieve(text: str, voice: str, genre: str = "general", k: int = 4, role: str | None = None,
             exclude_ids=()) -> list[dict]:
    bank = load_bank(voice, genre)
    q = describe_query(text, bank.fw_top, role)
    res = bank.query(q, k, exclude_ids)
    for r in res:
        r["query"] = {"role": q["role"], "function": q["function"], "n_words": q["n_words"]}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="source chunk file, or - for stdin")
    ap.add_argument("--text")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--genre", default="general")
    ap.add_argument("--role", choices=["intro", "methods", "results", "discussion", "other"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.text:
        text = a.text
    elif a.file == "-" or (a.file is None and not sys.stdin.isatty()):
        text = sys.stdin.read()
    elif a.file:
        doc = ex.extract(a.file)
        text = "\n\n".join(b.text for b in doc.paragraphs)
    else:
        ap.error("give FILE, --text, or text on stdin")
    res = retrieve(text, a.voice, a.genre, a.k, a.role)
    if a.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return
    q = res[0]["query"] if res else {}
    print(f"Query: role={q.get('role')} function={q.get('function')} ~{q.get('n_words')} words/paragraph\n")
    for i, r in enumerate(res, 1):
        print(f"--- Exemplar {i} [{r['role']}/{r['function']}, {r['n_words']} words, rel {r['relevance']}] "
              f"({r['source']})")
        print(r["text"] + "\n")


if __name__ == "__main__":
    main()
