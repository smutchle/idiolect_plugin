import math
from pathlib import Path

import pytest

import extract as ex
import score
import voiceprofile as vp
from conftest import CORPORA

AI_PARA = (
    "In today's fast-paced world, the role of public libraries has evolved significantly. No longer "
    "just repositories of books, libraries now serve as vibrant community hubs that offer a wide range "
    "of services, including digital literacy programs, job search assistance, and spaces for "
    "collaboration. Moreover, they play a crucial role in bridging the digital divide by providing free "
    "internet access to those who might otherwise be left behind. However, many libraries face "
    "significant challenges, such as budget cuts and declining foot traffic. It is important to "
    "recognize that libraries are not just about books; they are about people. Ultimately, the future "
    "of libraries will depend on their ability to adapt to changing needs while remaining true to "
    "their core mission. By investing in libraries, communities can foster lifelong learning, promote "
    "social inclusion, and ensure that everyone has access to the resources they need to thrive."
)


# ---------------------------------------------------------------- unit

def _pf(**kw):
    base = {"value": 1.0, "group": "punctuation", "kind": "rate", "scale": 1000.0, "between_sd": 0.5,
            "per_sample_n": [5000], "weight": 1.0, "reliability": "high"}
    base.update(kw)
    return base


def test_ai_tells_one_sided():
    pf = _pf(group="ai_tells", value=2.0)
    z_low, _ = score.feature_z("ai_vocab_rate", pf, 0.0, 1000)
    z_high, _ = score.feature_z("ai_vocab_rate", pf, 10.0, 1000)
    assert z_low == 0.0 and z_high > 2


def test_noise_shrinks_z_for_short_texts():
    pf = _pf(value=3.0)
    z_short, _ = score.feature_z("punct_semicolon", pf, 6.0, 200)
    z_long, _ = score.feature_z("punct_semicolon", pf, 6.0, 20000)
    assert z_short < z_long


def test_measurable_rules():
    pf = _pf(value=0.5)                              # 0.5/1k -> 0.1 expected in 200 words
    tf = {"value": 0.0, "n": 200, "count": 0}
    assert not score.measurable("punct_exclam", pf, tf, 200)[0]
    tf = {"value": 15.0, "n": 200, "count": 3}       # observed excess is measurable
    assert score.measurable("punct_exclam", pf, tf, 200)[0]
    assert not score.measurable("sent_len_p90", _pf(kind="stat"), {"value": 1, "n": 10, "count": 10}, 200)[0]


def test_gap_phrasing():
    prof = {"features": {"fw:but": {"value": 4.2}, "fw:yet": {"value": 1.1},
                         "sent_len_p10": {"value": 7}, "sent_len_p90": {"value": 38}}}
    r = {"value": 6.1, "target": 0.8}
    msg = score.phrase_gap("fw:however", r, prof, 1000)
    assert msg.startswith("'however' at 6.1/1k; target 0.8. Remove most instances")
    assert "'but' (4.2/1k)" in msg and "'yet'" in msg
    msg = score.phrase_gap("punct_semicolon", {"value": 0.0, "target": 3.2}, prof, 1000)
    assert msg.startswith("Semicolons at 0.0/1k; target 3.2.")
    msg = score.phrase_gap("passive_share", {"value": 0.31, "target": 0.12}, prof, 1000)
    assert msg.startswith("Passive voice at 31% of clauses; target 12% of clauses.")
    msg = score.phrase_gap("sent_len_sd", {"value": 4.1, "target": 9.7}, prof, 1000)
    assert "under 8 words and some over 38" in msg


def test_band_for_length_picks_nearest():
    bands = {"250": {"mean_words": 300, "composite": {"lo": 1, "median": 1.2, "hi": 1.5, "p95": 1.6},
                     "burrows_delta": {"hi": 3}},
             "1000": {"mean_words": 1100, "composite": {"lo": .8, "median": 1, "hi": 1.2, "p95": 1.3},
                      "burrows_delta": {"hi": 2}}}
    assert score.band_for_length(bands, 200)["length_bucket"] == "250"
    assert score.band_for_length(bands, 2000)["length_bucket"] == "1000"


# ---------------------------------------------------------------- integration

@pytest.fixture(scope="module")
def corpora():
    data = {}
    for a in ("emerson", "chesterton"):
        docs = [ex.extract(f) for f in sorted((CORPORA / a).glob("*.txt"))]
        pcs = [vp.paragraph_counts(d.paragraphs) for d in docs]
        data[a] = (pcs, [vp.finalize_counts(p) for p in pcs], [Path(d.path).name for d in docs])
    return data


@pytest.mark.slow
def test_two_author_leave_one_out(corpora):
    """Each held-out sample is closer to its own (LOO) profile than to the other author's."""
    full = {a: vp.build_profile(corpora[a][1]) for a in corpora}
    chunk_hits = chunk_n = 0
    for a, b in (("emerson", "chesterton"), ("chesterton", "emerson")):
        pcs, scs, _ = corpora[a]
        for i in range(len(scs)):
            own = vp.build_profile([c for j, c in enumerate(scs) if j != i])
            so, sb = score.score_counts(scs[i], own, 0), score.score_counts(scs[i], full[b], 0)
            assert so["composite"] < sb["composite"], (a, i)
            assert so["cosine_delta"] < sb["cosine_delta"], (a, i)
            assert so["burrows_delta"] < sb["burrows_delta"], (a, i)
            for ch in score.chunk_paragraph_counts(pcs[i], 1000):
                c = vp.finalize_counts(ch, prune=False)
                chunk_hits += score.score_counts(c, own, 0)["composite"] < score.score_counts(c, full[b], 0)["composite"]
                chunk_n += 1
    assert chunk_hits / chunk_n >= 0.9


@pytest.mark.slow
def test_calibration_band_and_ai_baseline(corpora):
    pcs, scs, names = corpora["chesterton"]
    prof = vp.build_profile(scs)
    cal = score.calibrate(pcs, scs, names, prof)
    b500 = cal["bands"]["500"]["composite"]
    assert 0 < b500["lo"] < b500["median"] < b500["hi"]
    # longer texts are measured more precisely -> the band tightens
    assert cal["bands"]["full"]["composite"]["median"] < b500["median"]
    # generic AI-style prose falls outside the within-author band
    assert cal["baseline"]["500"]["composite"] > b500["hi"]
    assert cal["baseline"]["full"]["composite"] > cal["bands"]["full"]["composite"]["max"]

    # sentence rhythm: AI-style uniformity is clearly outside the author's band at >=1000 words
    assert cal["baseline"]["1000"]["rhythm"] > cal["bands"]["1000"]["rhythm"]["hi"]
    assert cal["baseline"]["full"]["rhythm"] > cal["bands"]["full"]["rhythm"]["hi"]

    # a single ~150-word paragraph: honest about what it can't measure, still finds AI tells
    sc = score.score_counts(vp.counts_from_text(AI_PARA), prof, 8, cal)
    assert sc["short_text"] and sc["unmeasurable"]
    assert any(g["feature"].startswith("tell:") for g in sc["gaps"])


def test_rhythm_distance_detects_uniform_lengths():
    """Same median length, very different spread: the shape distance must see it."""
    import numpy as np
    rng = np.random.default_rng(0)
    author = [[int(x) for x in np.clip(rng.lognormal(np.log(20), 0.7, 8), 2, 120)] for _ in range(60)]
    prof = {"sentence_lengths": [author[i::4] for i in range(4)]}
    varied = {"sent_lens": [int(x) for x in np.clip(rng.lognormal(np.log(20), 0.7, 80), 2, 120)]}
    uniform = {"sent_lens": [int(x) for x in np.clip(rng.normal(20, 3, 80), 8, 40)]}
    rv, ru = score.rhythm_row(varied, prof), score.rhythm_row(uniform, prof)
    assert ru["z"] > 3 and rv["z"] < 2
    assert ru["bins"][3]["text"] < ru["bins"][3]["author"]        # too few 35+ sentences
    msg = score.phrase_gap("sent_len_w1", ru, {"features": {}}, 1600)
    assert "35+ words" in msg or "under 10 words" in msg


def test_log_space_z_for_spread():
    pf = {"value": 14.8, "group": "rhythm", "kind": "stat", "between_sd": 3.0, "between_sd_log": 0.15,
          "per_sample_n": [300], "log": True}
    z_log, _ = score.feature_z("sent_len_sd", pf, 6.4, 300)
    pf_lin = dict(pf); pf_lin.pop("between_sd_log")
    z_lin, _ = score.feature_z("sent_len_sd", pf_lin, 6.4, 300)
    assert z_log < z_lin < 0 and z_log < -3.5 and z_lin > -2.5


def test_rhythm_gap_leads_report():
    sc = {"words": 1000, "features": {
        "word_len_mean": {"value": 5.6, "target": 4.4, "z": 12.0, "weight": 1.0, "group": "lexical"},
        "sent_len_sd": {"value": 6.0, "target": 14.0, "z": -3.0, "weight": 1.0, "group": "rhythm"}}}
    prof = {"features": {"sent_len_p10": {"value": 7}, "sent_len_p90": {"value": 40}}}
    gaps = score.gap_report(sc, prof, 5)
    assert gaps[0]["feature"] == "sent_len_sd"
