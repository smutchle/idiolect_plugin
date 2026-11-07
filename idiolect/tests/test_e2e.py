"""End-to-end: an AI-written passage rewritten (by the model, following the skill) into
each author's voice. The rewrites are stored fixtures, produced by the full loop and passed by
the model fidelity pass; these tests check the measurable claims:
the distance to the target voice drops into the author's band, sentence rhythm moves toward
the author, and the deterministic fidelity check passes."""
import os

import pytest

import extract as ex
import fidelity as fd
import retrieve as rt
import score
import voiceprofile as vp
from conftest import CORPORA, FIXTURES

E2E = FIXTURES / "e2e"


@pytest.fixture(scope="module")
def voices(tmp_path_factory):
    home = tmp_path_factory.mktemp("idhome")
    old = os.environ.get("IDIOLECT_HOME")
    os.environ["IDIOLECT_HOME"] = str(home)
    for v in ("emerson", "chesterton"):
        vp.run_profile([CORPORA / v], v, "essay")
    rt.load_bank.cache_clear()
    yield {v: score.load_voice(v, "essay") for v in ("emerson", "chesterton")}
    if old is None:
        os.environ.pop("IDIOLECT_HOME", None)
    else:
        os.environ["IDIOLECT_HOME"] = old


@pytest.fixture(autouse=True)
def _isolated_home():
    yield


def _score(path, prof, cal):
    doc = ex.extract(path)
    c = vp.finalize_counts(vp.paragraph_counts(doc.paragraphs), doc.citations, prune=False)
    return score.score_counts(c, prof, 8, cal)


def _body(path):
    d = ex.extract(path, for_rewrite=True)
    return "\n\n".join(b.text for b in d.blocks if b.kind in ("para", "list"))


@pytest.mark.slow
@pytest.mark.parametrize("voice", ["emerson", "chesterton"])
def test_rewrite_moves_into_target_band_with_fidelity(voices, voice):
    prof, cal = voices[voice]
    src = E2E / "ai_nature.md"
    out = E2E / f"ai_nature.{voice}.md"
    s0, s1 = _score(src, prof, cal), _score(out, prof, cal)
    assert s1["composite"] < 0.85 * s0["composite"]
    assert s1["inside_band"] is True
    assert s1["group_scores"]["rhythm"] < 0.5 * s0["group_scores"]["rhythm"]
    assert s1["group_scores"].get("ai_tells", 0.0) < s0["group_scores"]["ai_tells"]
    fid = fd.compare_texts(_body(src), _body(out))
    assert fid["n_issues"] == 0, fid
