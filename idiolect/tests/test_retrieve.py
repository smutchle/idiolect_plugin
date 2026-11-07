import json
import math
import random

import pytest

import retrieve as rt
import voiceprofile as vp
from common import get_nlp, voice_dir
from conftest import CORPORA


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    import os
    home = tmp_path_factory.mktemp("idhome")
    old = os.environ.get("IDIOLECT_HOME")
    os.environ["IDIOLECT_HOME"] = str(home)
    vp.run_profile([CORPORA / "chesterton"], "gkc", "essay", calibrate=False)
    rt.load_bank.cache_clear()
    b = rt.load_bank("gkc", "essay")
    yield b
    rt.load_bank.cache_clear()
    if old is None:
        os.environ.pop("IDIOLECT_HOME", None)
    else:
        os.environ["IDIOLECT_HOME"] = old


@pytest.fixture(autouse=True)
def _isolated_home():          # override conftest: the module fixture manages IDIOLECT_HOME
    yield


def topic_swap(text: str) -> str:
    """Replace every content word with an unrelated one: same structure, new topic."""
    subs = {"NOUN": "widget", "PROPN": "Zorbia", "VERB": "calibrate", "ADJ": "purple"}
    out = []
    for t in get_nlp()(text):
        out.append((subs.get(t.pos_, t.text) if t.pos_ in subs else t.text) + t.whitespace_)
    return "".join(out)


def topic_only(text: str) -> str:
    """Same topic words, completely different structure (terse noun list)."""
    nouns = [t.text for t in get_nlp()(text) if t.pos_ in ("NOUN", "PROPN")][:30]
    return " ".join(f"{n.capitalize()} matters." for n in nouns)


@pytest.mark.slow
def test_structure_not_topic(bank):
    rng = random.Random(1)
    recs = [r for r in bank.recs if 80 <= r["n_words"] <= 250]
    hits_struct = hits_topic = 0
    sample = rng.sample(recs, 12)
    for r in sample:
        q = rt.describe_query(topic_swap(r["text"]), bank.fw_top, role=r["role"])
        ids = [x["id"] for x in bank.query(q, 4)]
        hits_struct += r["id"] in ids
        q2 = rt.describe_query(topic_only(r["text"]), bank.fw_top, role=r["role"])
        hits_topic += r["id"] in [x["id"] for x in bank.query(q2, 4)]
    assert hits_struct >= 9            # same structure, new topic: still finds the paragraph
    assert hits_topic <= 1             # same topic, new structure: does not


@pytest.mark.slow
def test_role_filter_and_diversity(bank):
    text = bank.recs[0]["text"]
    q = rt.describe_query(text, bank.fw_top, role="discussion")
    res = bank.query(q, 4)
    assert len(res) == 4
    assert all(r["role"] == "discussion" for r in res)       # enough discussion paragraphs exist
    assert len({r["id"] for r in res}) == 4
    assert len({r["text"] for r in res}) == 4


@pytest.mark.slow
def test_length_matching_beats_random(bank):
    rng = random.Random(0)
    recs = bank.recs
    got, base = [], []
    for r in rng.sample(recs, 20):
        q = rt.describe_query(r["text"], bank.fw_top, role=r["role"])
        res = bank.query(q, 4, exclude_ids=[r["id"]])
        got += [abs(math.log(x["n_words"] / r["n_words"])) for x in res]
        base += [abs(math.log(x["n_words"] / r["n_words"])) for x in rng.sample(recs, 4)]
    assert sum(got) / len(got) < 0.6 * sum(base) / len(base)


@pytest.mark.slow
def test_exclude_and_cli_json(bank, capsys):
    r0 = bank.recs[5]
    res = rt.retrieve(r0["text"], "gkc", "essay", k=3, exclude_ids=[r0["id"]])
    assert r0["id"] not in [x["id"] for x in res]
    rt.main(["--voice", "gkc", "--genre", "essay", "--k", "2", "--json", "--text", r0["text"]])
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 2 and {"text", "role", "function", "relevance", "match"} <= set(out[0])


def test_lone_chunk_role_defaults_to_other():
    q = rt.describe_query("Libraries have evolved significantly over the years. They do many things.",
                          ["the", "of"])
    assert q["role"] == "other"
    q = rt.describe_query("## Methods\n\nWe collected 40 samples and measured each twice.", ["the"])
    assert q["role"] == "methods"


@pytest.mark.slow
def test_exemplars_have_typical_author_rhythm(bank):
    """Retrieved exemplars model the author's sentence rhythm better than random picks."""
    import numpy as np
    rng = random.Random(3)
    cv = lambda r: r["sent_len_sd"] / max(r["sent_len_mean"], 1)
    med = float(np.median([cv(r) for r in bank.recs if r["n_sents"] >= 3]))
    got, base = [], []
    by_id = {r["id"]: r for r in bank.recs}
    for r in rng.sample(bank.recs, 20):
        res = bank.query(rt.describe_query(r["text"], bank.fw_top, r["role"]), 4, exclude_ids=[r["id"]])
        got += [abs(math.log(max(cv(by_id[x["id"]]), .05) / med)) for x in res]
        base += [abs(math.log(max(cv(x), .05) / med)) for x in rng.sample(bank.recs, 4)]
    assert np.mean(got) < 0.8 * np.mean(base)
