import math

import pytest

import extract as ex
import voiceprofile as vp

from conftest import CORPORA


def feats(text, fw=("the", "however", "but"), phrases=(), openers=("however,", "the")):
    c = vp.counts_from_text(text)
    return c, vp.compute_features(c, list(fw), list(phrases), list(openers))


def test_semicolons_and_passives_hand_checked():
    text = ("The samples were collected in May; the results were analyzed in June; "
            "nothing else happened; we went home.")
    c, f = feats(text)
    assert c["punct"]["semicolon"] == 3
    assert c["passive"] == 2
    assert c["words"] == 18
    assert f["punct_semicolon"]["value"] == pytest.approx(3 * 1000 / 18)
    assert f["passive_share"]["value"] == pytest.approx(2 / 4)   # 4 subject clauses


def test_sentence_lengths_and_shares():
    long = "The long sentence " + " ".join(["keeps going"] * 18) + " and then it ends"
    text = "It is short. " + long + ". Five words are right here."
    c, f = feats(text)
    assert c["sent_lens"] == [3, 43, 5]
    assert f["sent_len_mean"]["value"] == pytest.approx(17.0)
    assert f["sent_short_share"]["value"] == pytest.approx(2 / 3)
    assert f["sent_long_share"]["value"] == pytest.approx(1 / 3)
    assert f["sent_len_p50"]["value"] == pytest.approx(5)


def test_pronouns_contractions_questions_openers():
    text = ("However, I don't think we can go. You can't either, can you? "
            "The plan isn't ours. However, my view is clear!")
    c, f = feats(text)
    assert c["fp_sing"] == 2            # I, my
    assert c["fp_plur"] == 2            # we, ours
    assert c["second"] == 2             # You, you
    assert c["contraction"] == 3        # n't x3
    assert c["punct"]["question"] == 1 and c["punct"]["exclam"] == 1
    assert c["opener_word"]["however,"] == 2
    assert f["opener:however,"]["value"] == pytest.approx(2 / 4)


def test_function_word_rates_and_discourse_tracking():
    text = "The cat sat on the mat. But the dog did not. However, the bird sang."
    c, f = feats(text)
    W = c["words"]
    assert f["fw:the"]["value"] == pytest.approx(4 * 1000 / W)
    assert f["fw:however"]["count"] == 1
    sel = vp.select_function_words(c, k=5)
    assert "moreover" in sel            # discourse words always tracked, even unused


def test_ai_tells_counted():
    text = ("This is not just a tool, but a revolution. We must delve into the rich tapestry "
            "of ideas. The result: a better world. Is it worth it? Absolutely. It matters a lot.\n\n"
            "Teams need speed, clarity, and trust. Ultimately, this is crucial.")
    c, f = feats(text)
    t = c["tells"]
    assert t["s01"] == 1                # not just X, but Y
    assert t["s12"] == 3                # delve, tapestry, crucial
    assert t["s02"] >= 1                # colon reveal
    assert t["s04"] >= 1                # question answered at once
    assert t["s06"] >= 1                # triad
    assert t["s13"] == 1                # 'Ultimately,' send-off
    assert f["tell:s12"]["group"] == "ai_tells"


def test_mattr():
    assert vp.mattr(["a", "b", "c", "d"]) == 1.0
    toks = ["a", "b"] * 600
    assert vp.mattr(toks, window=500) == pytest.approx(2 / 500)


def test_ngram_topic_guard():
    text = ("In other words, the neural network learned. In other words, the gradient vanished. "
            "In other words, the transformer converged.")
    c, _ = feats(text)
    assert c["ngrams"]["in other words"] == 3
    assert not any("network" in k or "gradient" in k for k in c["ngrams"])


def test_counts_are_additive():
    a = vp.counts_from_text("First text here. It is short; yes.")
    b = vp.counts_from_text("Second text. Also short, but fine.")
    m = vp.merge_counts([a, b])
    assert m["words"] == a["words"] + b["words"]
    assert m["punct"]["semicolon"] == 1
    rt = vp.counts_from_json(vp.counts_to_json(m))
    assert rt["tok"] == m["tok"] and rt["sent_lens"] == m["sent_lens"]


def test_reliability_flags():
    f = {"kind": "rate", "count": 3, "n": 50000}
    assert vp.reliability(f, "punct_exclam", 50000) == ("low", 0.0)
    f = {"kind": "rate", "count": 400, "n": 50000}
    assert vp.reliability(f, "fw:the", 50000) == ("high", 1.0)
    # avoiding a word general English uses often is informative in a large corpus
    f = {"kind": "rate", "count": 0, "n": 50000}
    assert vp.reliability(f, "fw:however", 50000)[0] == "medium"


def test_pseudo_samples_when_few_files():
    paras = [ex.Block(" ".join(["word"] * 100) + ".") for _ in range(40)]
    doc = ex.Document("one.txt", blocks=paras)
    groups, names, _, pseudo = vp.make_samples([doc])
    assert pseudo and len(groups) == vp.PSEUDO_SAMPLES
    assert sum(len(g) for g in groups) == 40


def test_small_files_are_grouped():
    docs = [ex.Document(f"f{i}.md", blocks=[ex.Block(" ".join(["w"] * 600) + ".")]) for i in range(9)]
    groups, names, _, pseudo = vp.make_samples(docs)
    assert not pseudo and len(groups) == 3
    assert names[0] == "f0.md+f1.md+f2.md"


@pytest.mark.slow
def test_profile_directory_end_to_end(tmp_path):
    import json
    from common import voice_dir
    prof = vp.run_profile([CORPORA / "chesterton"], "gkc", "essay")
    vd = voice_dir("gkc", "essay")
    for name in ("profile.json", "samples.json", "exemplars.jsonl", "sources.txt"):
        assert (vd / name).exists()
    assert prof["corpus"]["samples"] == 8 and prof["corpus"]["words"] > 50000
    f = prof["features"]
    assert f["fw:the"]["reliability"] == "high" and len(f["fw:the"]["per_sample"]) == 8
    assert f["fw:the"]["between_sd"] > 0
    assert len(prof["selection"]["phrases"]) == vp.N_PHRASES
    ex_recs = [json.loads(l) for l in (vd / "exemplars.jsonl").open(encoding="utf-8")]
    assert all(r["n_words"] >= 40 for r in ex_recs)
    assert {"role", "function", "masked", "fw", "text"} <= set(ex_recs[0])
    s = vp.summarize(prof)
    assert "Most distinctive habits" in s


CASUAL = ("I can't believe it worked. You'd think I'd have learned by now, but I didn't. "
          "We tried again and it's fine. Honestly, you won't regret it! ")
ACADEMIC = ("The measurements were obtained using a calibrated instrument, and the observations "
            "were subsequently normalized with respect to the baseline condition. The resulting "
            "distributions were evaluated by means of a nonparametric procedure; differences were "
            "considered significant at the conventional threshold. ")


def test_genre_mixing_warning():
    samples = [vp.counts_from_text(CASUAL * 25) for _ in range(2)] + \
              [vp.counts_from_text(ACADEMIC * 12) for _ in range(2)]
    prof = vp.build_profile(samples, {"samples": ["c1", "c2", "a1", "a2"]})
    assert any("mix genres" in w for w in prof["warnings"])
    same = vp.build_profile([vp.counts_from_text(CASUAL * 25) for _ in range(3)])
    assert not any("mix genres" in w for w in same["warnings"])
