import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

import extract as ex
import retrieve as rt
import rewrite as rw
import voiceprofile as vp
from conftest import CORPORA

PAPER = """# A Study

## Introduction

Prior work has looked at this problem [3]. Others disagree (Smith et al., 2020). The effect was large and it was measured with care across many different settings over several years of work.

Figure 1: A plot of things.

## Methods

We collected 120 samples and measured each one twice, using `numpy` and the protocol in [4]. This took about 3 weeks and cost $2,000 in total, which we considered reasonable for a study of this size.

$$ y = \\alpha + \\beta x $$

| a | b |
|---|---|
| 1 | 2 |

## References

[3] A. Author. Some paper. 2019.
"""


@pytest.fixture(scope="module")
def voice(tmp_path_factory):
    home = tmp_path_factory.mktemp("idhome")
    old = os.environ.get("IDIOLECT_HOME")
    os.environ["IDIOLECT_HOME"] = str(home)
    vp.run_profile([CORPORA / "chesterton"], "gkc", "essay")
    rt.load_bank.cache_clear()
    yield home
    rt.load_bank.cache_clear()
    if old is None:
        os.environ.pop("IDIOLECT_HOME", None)
    else:
        os.environ["IDIOLECT_HOME"] = old


@pytest.fixture(autouse=True)
def _isolated_home():
    yield


def test_rewrite_mode_keeps_apparatus(tmp_path):
    p = tmp_path / "paper.md"
    p.write_text(PAPER, encoding="utf-8")
    doc = ex.extract(p, for_rewrite=True)
    kinds = [b.kind for b in doc.blocks]
    assert "protected" in kinds
    md = rw.assemble_markdown(doc.blocks, {})
    for must in ("[3]", "(Smith et al., 2020)", "`numpy`", "Figure 1: A plot of things.", "\\alpha",
                 "| 1 | 2 |", "[3] A. Author. Some paper. 2019.", "## References", "$2,000"):
        assert must in md, must
    prof_doc = ex.extract(p)     # profiling mode still strips all of it
    assert "Some paper" not in prof_doc.text and "[3]" not in prof_doc.text


def test_chunking_rules():
    blocks = [ex.Block("Intro", "heading", 1)]
    blocks += [ex.Block(" ".join(["word"] * 120) + ".") for _ in range(5)]
    blocks += [ex.Block("Methods", "heading", 1), ex.Block(" ".join(["w"] * 60) + ".")]
    blocks += [ex.Block("x = 1", "protected"), ex.Block(" ".join(["v"] * 500) + ".")]
    chunks = rw.make_chunks(blocks)
    for ch in chunks:
        assert all(blocks[i].kind in ("para", "list") for i in ch["blocks"])
        heads = {blocks[i].heading for i in ch["blocks"]}
        assert len(heads) <= 1
    sizes = [c["words"] for c in chunks]
    assert sizes[0] >= rw.CHUNK_MIN and all(s <= rw.CHUNK_MAX or c["paragraphs"] == 1
                                            for s, c in zip(sizes, chunks))
    assert any(c["paragraphs"] == 1 and c["words"] > 400 for c in chunks)   # never split a paragraph


@pytest.mark.slow
def test_prepare_assemble_finalize_md(voice, tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(PAPER, encoding="utf-8")
    res = rw.cmd_prepare(Namespace(source=str(src), voice="gkc", genre="essay", rounds=1, output=None,
                                   run_dir=str(tmp_path / "run")))
    run = Path(res["run_dir"])
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert res["chunks"] == len(st["chunks"]) >= 2
    p1 = (run / "prompts" / "r1" / "c001.md").read_text(encoding="utf-8")
    for section in ("## Style brief", "## Rhythm target", "## Exemplars", "## Hard constraints", "## SOURCE"):
        assert section in p1
    assert "{" not in p1.split("## Style brief")[1].split("## Rhythm")[0]   # no raw JSON
    # incomplete: nothing written yet
    out = rw.cmd_assemble(Namespace(run_dir=str(run)))
    assert out["status"] == "incomplete" and out["missing"]
    # paragraph-count mismatch is rejected
    for ch in st["chunks"]:
        (run / "rewrites" / f"{ch['id']}.md").write_text("one.\n\ntwo.\n\nthree.\n\nfour.", encoding="utf-8")
    out = rw.cmd_assemble(Namespace(run_dir=str(run)))
    assert out["status"] == "incomplete" and out["paragraph_mismatch"]
    # identity "rewrite" -> scores like the source; rounds=1 -> stops after one revision round at most
    for ch in st["chunks"]:
        (run / "rewrites" / f"{ch['id']}.md").write_text((run / "chunks" / f"{ch['id']}.source.md").read_text(encoding="utf-8"), encoding="utf-8")
    out = rw.cmd_assemble(Namespace(run_dir=str(run)))
    assert out["round"] == 1 and out["status"] in ("revise", "stop")
    if out["status"] == "revise":
        assert out["revise_chunks"]
        out = rw.cmd_assemble(Namespace(run_dir=str(run)))
        assert out["status"] == "stop"
    fin = rw.cmd_finalize(Namespace(run_dir=str(run)))
    text = Path(fin["output"]).read_text(encoding="utf-8")
    assert "[3] A. Author. Some paper. 2019." in text and "$$ y" in text and "| 1 | 2 |" in text
    assert "NOT RUN" in fin["scorecard_md"]
    assert Path(fin["scorecard"]).exists()


@pytest.mark.slow
def test_docx_in_place(voice, tmp_path):
    import docx
    d = docx.Document()
    d.add_heading("Results", level=1)
    para = "The model improved accuracy by 4.2% on the held-out set, which we did not expect. " * 3
    d.add_paragraph(para)
    d.add_paragraph(para.replace("4.2%", "5.1%"))
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "keep me"
    src = tmp_path / "x.docx"
    d.save(src)
    res = rw.cmd_prepare(Namespace(source=str(src), voice="gkc", genre="essay", rounds=0, output=None,
                                   run_dir=str(tmp_path / "run")))
    run = Path(res["run_dir"])
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    for ch in st["chunks"]:
        txt = (run / "chunks" / f"{ch['id']}.source.md").read_text(encoding="utf-8").replace("which we did not expect",
                                                                              "which we had not expected")
        (run / "rewrites" / f"{ch['id']}.md").write_text(txt, encoding="utf-8")
    assert rw.cmd_assemble(Namespace(run_dir=str(run)))["status"] == "stop"   # rounds=0
    fin = rw.cmd_finalize(Namespace(run_dir=str(run)))
    out = docx.Document(fin["output"])
    body = "\n".join(p.text for p in out.paragraphs)
    assert "had not expected" in body and "4.2%" in body and "5.1%" in body
    assert out.tables and out.tables[0].cell(0, 0).text == "keep me"
    assert out.paragraphs[0].style.name.startswith("Heading")
