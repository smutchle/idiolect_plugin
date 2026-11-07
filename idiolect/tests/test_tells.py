"""Detectors for the humanizer catalog: each pattern fires on upstream's 'Before' examples and the
'After' rewrites are (nearly) clean; plain human prose is not flagged."""
from argparse import Namespace
from pathlib import Path

import pytest

import humanize as hz
import rewrite as rw
import tells
from conftest import FIXTURES


def pats(text):
    hits, _ = tells.scan_text(text)
    return {h.pattern for h in hits}


BEFORE = {
    "s01": "It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere.",
    "s02": "Caching cuts repeat work and saves a great deal of time across every request the service handles during "
           "peak hours, which matters for cost.\n\nThat is the real win.",
    "s03": "The real question is whether teams can adapt. At its core, what really matters is readiness.",
    "s04": "Let's dive into how caching works in Next.js. Here's what you need to know.",
    "s05": "This isn't mainly about prompt length, and I'm not arguing that documentation doesn't matter.",
    "s06": "Attendees can expect innovation, inspiration, and industry insights.",
    "s07": "She noted the door. She noted the lock on it. She filed both away.",
    "s08": "The new policy — announced without warning — affects thousands of workers.",
    "s09": "It could potentially be argued that the policy might have some effect on outcomes.",
    "s10": "The report is high-quality and the plan is long-term.",
    "s12": "Additionally, the enduring testament to this is the pivotal role of camel meat.",
    "s13": "The institute was established in 1989, marking a pivotal moment in the evolution of statistics.",
    "s14": "He is associated with the Rajhans Orchestra.",
    "s15": "The temple uses blue and gold, symbolizing the bluebonnets of Texas.",
    "s16": "Nestled within the breathtaking region of Gonder, the town has stunning natural beauty.",
    "s17": "Experts believe it plays a role in the regional ecosystem.",
    "s18": "Gallery 825 serves as the exhibition space and boasts over 3,000 square feet.",
    "s19": "- **User Experience:** The interface was improved.\n- **Performance:** It is faster.",
    "s20": "## Strategic Negotiations And Global Partnerships\n\nThe firm signed two agreements in 2023 with suppliers.",
    "s22": "Great question! The French Revolution began in 1789. I hope this helps!",
    "s23": "While specific details are limited, it appears to have been established in the 1990s.",
    "s24": "## Performance\n\nPerformance matters.\n\nWhen users hit a slow page, they leave.",
    "s25": "The table below compares the six options by price.",
}


@pytest.mark.parametrize("pid", sorted(BEFORE))
def test_pattern_fires_on_upstream_example(pid):
    assert pid in pats(BEFORE[pid]), (pid, pats(BEFORE[pid]))


def test_after_examples_are_clean():
    afters = ["The heavy beat adds to the aggressive tone.",
              "Gallery 825 is LAAA's exhibition space for contemporary art. The gallery has four rooms totaling "
              "3,000 square feet.",
              "The French Revolution began in 1789 when a financial crisis and food shortages led to unrest.",
              "Next.js caches data at multiple layers, including request memoization and the router cache."]
    for a in afters:
        assert not (pats(a) - tells.WEAK), (a, pats(a))


def test_relative_clause_is_not_a_triad():
    assert "s06" not in pats("Code generation is much faster, which reduces friction and gives developers more power.")


def test_human_prose_has_few_strong_tells():
    text = (FIXTURES / "corpora" / "chesterton" / "chesterton_01.txt").read_text()[:20000]
    hits, words = tells.scan_text(text)
    strong = sum(h.pattern in tells.STRONG for h in hits)
    assert 1000 * strong / words < 3.0
    ai, aw = tells.scan_file(FIXTURES / "e2e" / "ai_nature.md")
    assert 1000 * len(ai) / aw > 3 * 1000 * len(hits) / words


def test_wrapper_only_paragraphs_are_cuttable(tmp_path):
    src = tmp_path / "x.md"
    src.write_text("Great question! I hope this helps!\n\nThe study enrolled 40 people in 2020 and followed them "
                   "for a year, measuring sleep every week with a wrist sensor and a diary.\n\nLet me know if you "
                   "want more!\n")
    res = hz.cmd_prepare(Namespace(source=str(src), rounds=1, allow_cuts=False, output=None,
                                   run_dir=str(tmp_path / "run")))
    run = Path(res["run_dir"])
    import json
    st = json.loads((run / "state.json").read_text())
    ch = st["chunks"][0]
    assert ch["wrapper_only"] == [0, 2]
    assert "Paragraph 1 is entirely chatbot residue" in (run / "chunks" / "c001.tells.md").read_text()
    # cutting a content paragraph without --allow-cuts is rejected
    (run / "rewrites" / "c001.md").write_text("[CUT]\n\n[CUT]\n\n[CUT]\n")
    assert hz.cmd_assemble(Namespace(run_dir=str(run)))["status"] == "incomplete"
    (run / "rewrites" / "c001.md").write_text(
        "[CUT]\n\nThe study enrolled 40 people in 2020 and followed them for a year, measuring sleep every week "
        "with a wrist sensor and a diary.\n\n[CUT]\n")
    out = hz.cmd_assemble(Namespace(run_dir=str(run)))
    assert out["status"] == "stop" and out["tells"]["strong"] == 0
    fin = hz.cmd_finalize(Namespace(run_dir=str(run)))
    text = Path(fin["output"]).read_text()
    assert "hope this helps" not in text and "40 people" in text and "[CUT]" not in text
    assert "FIDELITY NOT CHECKED" in fin["report_md"]


def test_tells_section_respects_author_habits():
    hits, words = tells.scan_text("The plan — a good one — worked. The team — all of it — agreed. "
                                  "It was, in short, a crucial step.")
    sec = rw.tells_section(hits, words, {"s08": 4.0}, allow_cuts=False)
    assert "author habit" in sec and "s12" in sec


def test_humanize_cli_defaults_to_cuts(tmp_path, capsys):
    src = tmp_path / "y.md"
    src.write_text("In today's fast-paced world, the study enrolled 40 people. Ultimately, it matters.\n")
    hz.main(["prepare", str(src), "--run-dir", str(tmp_path / "r1")])
    import json
    assert json.loads((tmp_path / "r1" / "state.json").read_text())["allow_cuts"] is True
    hz.main(["prepare", str(src), "--strict", "--run-dir", str(tmp_path / "r2")])
    assert json.loads((tmp_path / "r2" / "state.json").read_text())["allow_cuts"] is False
    capsys.readouterr()
