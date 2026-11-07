# idiolect

A Claude Code plugin that:

1. **learns a writing voice** (yours, or one you have permission to use) from a folder of documents,
2. **rewrites documents in that voice** without changing what they say, and
3. **removes AI-writing tells** (the [blader/humanizer](https://github.com/blader/humanizer) catalog),
   with or without a voice.

The main use is turning AI-drafted text into your own voice.

**How it works:** deterministic Python measures style (sentence rhythm, function words,
punctuation, syntax, AI tells); the model rewrites using real passages from the author as
exemplars; Python re-measures the whole document; the model revises against specific numeric gaps;
a separate fidelity pass checks that no claim, number, citation or name changed. The model is never
asked to judge statistics itself.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Commands and examples](#commands-and-examples)
  - [/idiolect:voice-profile](#idiolectvoice-profile--build-a-voice)
  - [/idiolect:voice-list](#idiolectvoice-list--see-your-voices)
  - [/idiolect:voice-rewrite](#idiolectvoice-rewrite--rewrite-documents-in-a-voice)
  - [/idiolect:humanize](#idiolecthumanize--remove-ai-tells-without-a-voice)
  - [Asking in plain English](#asking-in-plain-english)
- [Cuts: how much may be removed](#cuts-how-much-may-be-removed)
- [Where files go](#where-files-go)
- [Reading the scorecard and the humanize report](#reading-the-scorecard-and-the-humanize-report)
- [Command-line tools](#command-line-tools)
- [Corpus advice](#corpus-advice)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Appropriate use](#appropriate-use) · [Credits](#credits) · [Development](#development)

---

## Install

### Prerequisites

- Claude Code (CLI, desktop or IDE extension).
- Python 3.10 or newer, and either [`uv`](https://docs.astral.sh/uv/) (recommended) or the standard
  `venv` + `pip`.
- Network access once, for the setup step. After that nothing calls the network or a paid API.

### Option A: install from GitHub (recommended)

```bash
claude plugin marketplace add smutchle/idiolect_plugin
claude plugin install idiolect@idiolect-local --scope user
bash ~/.claude/plugins/cache/idiolect-local/idiolect/*/setup.sh
```

The first two lines add the marketplace and install the plugin (`--scope user` makes it available
in every project; `--scope project` limits it to the current one). The third creates the Python
environment (see Step 1 below for what it does). Then continue with [Step 3](#step-3-load-it).

### Option B: install from a local clone

```bash
git clone https://github.com/smutchle/idiolect_plugin.git /path/to/idiolect_plugin
```

The repository is a small local marketplace holding the plugin:

```
idiolect_plugin/                     <- the marketplace (add this folder)
├── .claude-plugin/marketplace.json
└── idiolect/                        <- the plugin
    ├── .claude-plugin/plugin.json
    ├── setup.sh
    ├── scripts/  skills/  tests/
    └── README.md
```

In the commands below, replace `/path/to/idiolect_plugin` with where that folder is on your machine.

#### Step 1: create the Python environment (once)

```bash
bash /path/to/idiolect_plugin/idiolect/setup.sh
```

This creates `~/.idiolect/venv`, installs spaCy, PyMuPDF, python-docx, numpy, scikit-learn and
wordfreq, and downloads the spaCy English model. It ends with
`idiolect: environment ready at ~/.idiolect/venv`. Running it again is safe.

#### Step 2: install the plugin

```bash
claude plugin marketplace add /path/to/idiolect_plugin
claude plugin install idiolect@idiolect-local --scope user
```

Alternative, without installing (one session only):

```bash
claude --plugin-dir /path/to/idiolect_plugin/idiolect
```

### Step 3: load it

- In a Claude Code session that is already open, type `/reload-plugins`.
- New sessions load it automatically. No restart is needed.

### Step 4: check that it is active

- Type `/idiolect:` in the prompt. Autocomplete lists `voice-profile`, `voice-rewrite`,
  `voice-list`, `humanize` and `idiolect`.
- Or run `/idiolect:voice-list`. With no voices yet it says:
  `No voices yet in ~/.idiolect/voices. Create one with /idiolect:voice-profile.`
- Or, in a shell: `claude plugin list` (shows `idiolect@idiolect-local`, enabled), or
  `claude plugin details idiolect@idiolect-local` (lists the 5 skills and their context cost).

### Updating

To get a newer version (installed from GitHub or a local clone):

```bash
claude plugin marketplace update idiolect-local
claude plugin update idiolect@idiolect-local
```

then `/reload-plugins` in open sessions.

If you edit a local clone yourself: Claude Code runs a cached copy of the plugin (in
`~/.claude/plugins/cache/`), so edits do not take effect by themselves. Bump `"version"` in
`idiolect/.claude-plugin/plugin.json`, then:

```bash
claude plugin marketplace update idiolect-local
claude plugin update idiolect@idiolect-local
```

and `/reload-plugins` in open sessions. If an update does not pick up a change, reinstall:

```bash
claude plugin uninstall idiolect@idiolect-local
claude plugin install idiolect@idiolect-local --scope user
```

### Uninstalling

```bash
claude plugin uninstall idiolect@idiolect-local
claude plugin marketplace remove idiolect-local
```

Your voices and run history in `~/.idiolect/` are not touched; delete that folder to remove them
(and the Python environment).

---

## Quick start

```text
/idiolect:voice-profile me ~/writing/papers --genre paper
/idiolect:voice-rewrite me/paper ~/drafts/intro_from_chatgpt.md
/idiolect:humanize ~/drafts/blog_post.md
```

1. The first command reads every .pdf/.docx/.md/.txt under `~/writing/papers`, builds the voice
   `me` for the genre `paper`, and prints your most distinctive habits.
2. The second rewrites the draft in that voice and writes `~/drafts/intro_from_chatgpt.me.md`
   plus `intro_from_chatgpt.me.scorecard.md`.
3. The third removes AI tells from a post without using any voice and writes
   `blog_post.humanized.md` plus `blog_post.humanized.report.md`.

Claude does the writing in rounds and shows you the scorecard or report at the end. A typical
document takes a few minutes.

---

## Commands and examples

### `/idiolect:voice-profile` — build a voice

```text
/idiolect:voice-profile <name> <folder-or-files...> [--genre <genre>] [--add]
```

| Argument | Meaning |
|---|---|
| `<name>` | Voice name: letters, digits, `.`, `_`, `-` (e.g. `me`, `scott`, `team-style`). |
| `<folder-or-files...>` | One or more folders (searched recursively) and/or files: `.pdf`, `.docx`, `.md`, `.txt`. Hidden folders are skipped. |
| `--genre <genre>` | Default `general`. Keep genres separate: `paper`, `blog`, `email`, `grant`, `report`... A voice can have several genres. |
| `--add` | Add these documents to the existing voice/genre and rebuild it (earlier sources are kept). |

Examples:

```text
# All of your papers in one folder
/idiolect:voice-profile me ~/writing/papers --genre paper

# Specific files, mixed formats
/idiolect:voice-profile me ~/cv/paper1.pdf ~/cv/paper2.docx ~/notes/essay.md --genre paper

# Several folders at once
/idiolect:voice-profile me ~/writing/2023 ~/writing/2024 --genre blog

# A second genre for the same voice
/idiolect:voice-profile me ~/sent-emails-export --genre email

# Later: grow the paper voice with new work
/idiolect:voice-profile me ~/writing/new-papers --genre paper --add
```

What you get back (abridged, illustrative numbers):

```text
Corpus: 41,230 words, 1,912 sentences, 388 paragraphs, 9 samples, from 9 file(s).
Sentence length: mean 23.4, SD 13.1, median 21 (p10 9, p90 41); short 9%, long 17%.

Most distinctive habits (vs. generic/AI-style prose)
- semicolons per 1k words: 6.2 (generic prose: 0.7)
- 'we' 14.0/1k vs 2.8/1k in general English (more often)
- ...
Calibration (leave-one-out): band at ~1000 words 0.70-1.15; generic AI-style text lands at 2.30
Warnings: ...
```

The voice is stored in `~/.idiolect/voices/<name>/<genre>/` and persists across plugin updates.
For papers, references, captions, equations, affiliations, acknowledgments, headers/footers and
footnote markers are stripped automatically before measuring (the log says what was removed from
each file).

### `/idiolect:voice-list` — see your voices

```text
/idiolect:voice-list
```

```text
| voice | genre | words | samples | files | calibrated | band @1000w | created |
|---|---|---|---|---|---|---|---|
| me | email | 12,040 (low: <20k) | 6 | 214 | yes | 0.81-1.32 | 2026-11-07 |
| me | paper | 41,230 | 9 | 9 | yes | 0.70-1.15 | 2026-11-07 |
```

### `/idiolect:voice-rewrite` — rewrite documents in a voice

```text
/idiolect:voice-rewrite <voice>[/<genre>] <files-or-folders...> [--rounds N] [--allow-cuts] [--output-dir DIR]
```

| Argument | Meaning |
|---|---|
| `<voice>[/<genre>]` | e.g. `me/paper`. If the voice has only one genre, `me` is enough. |
| `<files-or-folders...>` | Documents to rewrite. Folders are expanded to their .md/.txt/.docx/.pdf files (earlier outputs are skipped). With more than 5 files Claude confirms the list first. |
| `--rounds N` | Maximum revision rounds after the first rewrite (default 3). |
| `--allow-cuts` | Also cut empty AI puffery (see [Cuts](#cuts-how-much-may-be-removed)). Off by default: the content of every sentence is kept. |
| `--output-dir DIR` | Write outputs there instead of next to each source. |

Examples:

```text
# One document
/idiolect:voice-rewrite me/paper drafts/intro_ai.md

# Several documents, any mix of formats
/idiolect:voice-rewrite me/paper drafts/intro_ai.md drafts/methods_ai.docx drafts/notes.txt

# Every document in a folder
/idiolect:voice-rewrite me/paper drafts/

# A puffery-heavy AI draft: also remove empty significance claims and send-offs
/idiolect:voice-rewrite me/paper drafts/ai_heavy.md --allow-cuts

# More revision rounds, outputs collected elsewhere
/idiolect:voice-rewrite me/blog posts/ --rounds 5 --output-dir posts/rewritten

# A different genre of the same voice
/idiolect:voice-rewrite me/email drafts/reply_to_reviewer.md
```

What happens, per document:

1. The source is extracted. References, captions, equations, tables, code and front matter are
   protected (passed through verbatim); citations, numbers and markdown inside paragraphs are kept.
2. It is split into 150-400-word chunks (never splitting a paragraph). For each chunk Claude gets a
   plain-English style brief, 4 of the author's paragraphs as exemplars (matched on structure, never
   on topic), a sentence-rhythm target, and the AI tells found in that passage, quoted, with the fix.
   Tells the author genuinely uses (per the profile) are kept at the author's rate.
3. Claude rewrites each chunk; the whole document is assembled and scored against the profile.
4. The biggest remaining gaps are traced to the chunks that cause them and revised (prompts show each
   passage's actual sentence lengths). It stops when the document is inside the author's own range,
   after `--rounds` rounds, or when a round improves less than 5%. The best version is kept, not the
   latest.
5. Fidelity check: numbers, units, citations and names are compared mechanically, then a separate
   reviewer lists any claim added, dropped or changed in strength. Flagged passages are fixed and
   re-checked.
6. Output in the source's format, plus a scorecard. `.docx` output is the original file edited in
   place, so styles, tables and images survive. `.pdf` input produces `.md` output.

Output for `drafts/intro_ai.md` with voice `me`: `drafts/intro_ai.me.md` and
`drafts/intro_ai.me.scorecard.md`.

### `/idiolect:humanize` — remove AI tells without a voice

```text
/idiolect:humanize <files-or-folders...> [--strict] [--rounds N] [--output-dir DIR] [--report-only]
```

| Argument | Meaning |
|---|---|
| `<files-or-folders...>` | Documents to clean up (.md/.txt/.docx/.pdf). |
| `--strict` | Fidelity first: reword tells, never cut content (see [Cuts](#cuts-how-much-may-be-removed)). By default humanize cuts empty puffery. |
| `--rounds N` | Maximum revision rounds (default 2). |
| `--output-dir DIR` | Write outputs there instead of next to each source. |
| `--report-only` | Only list the tells; change nothing. |

Examples:

```text
# See which tells a document has, without changing it
/idiolect:humanize drafts/post.md --report-only

# Clean up one document (cuts empty puffery, keeps every fact)
/idiolect:humanize drafts/post.md

# Reword only; never cut content
/idiolect:humanize drafts/grant_section.docx --strict

# A whole folder, outputs elsewhere
/idiolect:humanize drafts/ --output-dir drafts/clean
```

Chunks with no tells and varied sentence lengths are left untouched. The rest are rewritten
following humanizer's process: fix every listed tell, self-audit for tells that survive rewrites,
check that nothing was added or dropped, vary sentence length. The whole document is re-scanned
after each round; it stops when no strong or normal tells remain, after `--rounds` rounds, or when
a round removes less than 10% of the remaining tells. Then the same fidelity check and output
formats as voice rewrites. No opinions or anecdotes are added: voice comes from the kind of text
(neutral for technical writing).

Output for `drafts/post.md`: `drafts/post.humanized.md` and `drafts/post.humanized.report.md`.

If you have a voice for this kind of text, prefer `/idiolect:voice-rewrite`: it removes the same
tells, keeps the ones you genuinely use, and matches your voice.

### Asking in plain English

The skills also trigger from ordinary requests, for example:

- "Build a voice called `me` from the papers in `~/writing/papers`."
- "Rewrite `drafts/intro_ai.md` in my paper voice."
- "Which AI tells are in `drafts/post.md`?"
- "Clean the AI-isms out of `notes/summary.docx` but don't cut anything."
- "What voices do I have?"

---

## Cuts: how much may be removed

| Policy | What is removed | Default for | Switch |
|---|---|---|---|
| **Fidelity first** | Only chatbot residue ("Great question!", "I hope this helps") and staged run-ups ("Let's dive in"). Every other tell is reworded, with every claim kept at its strength. | `voice-rewrite` (your own drafts) | `voice-rewrite --allow-cuts` turns cuts on |
| **Cuts** | Also empty content: significance and sales dressing ("pivotal", "groundbreaking", "the future looks bright"), restating closers, send-off paragraphs, ", highlighting ..." riders, synonym cycling, knowledge-limit disclaimers. | `humanize` | `humanize --strict` turns cuts off |

Facts, numbers, names, citations and the items of real lists are kept under both policies. The
fidelity reviewer labels every removal as `allowed_cut` (listed, not a failure) or `dropped` (a
failure). Under fidelity first, puffery-heavy AI text keeps its (reworded) puffery, because the
fidelity check treats "groundbreaking" or "the ecosystem continues to thrive" as claims.

Example, on humanizer's own 306-word worst case: fidelity first took it from 61 tells to 5; cuts
took it from 61 to 6, with 13 removals labelled `allowed_cut` and no meaning changes.

---

## Where files go

| What | Where |
|---|---|
| Voices | `~/.idiolect/voices/<name>/<genre>/` (`profile.json`, `samples.json`, `exemplars.jsonl`, `calibration.json`, `sources.txt`) |
| Rewritten document | Next to the source (or `--output-dir`): `<stem>.<voice>.<ext>`; humanize: `<stem>.humanized.<ext>` |
| Scorecard / report | `<stem>.<voice>.scorecard.md`; humanize: `<stem>.humanized.report.md` |
| Working files for each run | `~/.idiolect/runs/<stem>-<voice>-<genre>-<timestamp>/` or `<stem>-humanize-<timestamp>/` (prompts, every round, scores, fidelity results) |
| Python environment | `~/.idiolect/venv` |

Set `IDIOLECT_HOME` to keep all of this somewhere other than `~/.idiolect`.

---

## Reading the scorecard and the humanize report

Voice-rewrite scorecard:

```text
| | composite | sentence rhythm | Burrows' Delta | Cosine Delta |
| source | 2.00 | 1.61 | 3.08 | 0.92 |
| output | 1.18 | 0.43 | 2.70 | 0.69 |
| author band (p10-p90, ~1156 words) | 0.76-1.20 | <= 1.18 | <= 2.56 | |

Composite: INSIDE the within-author band. Sentence rhythm: INSIDE.
Sentence lengths (output vs author): under 10 words 14% / 11%, 10-19 words 30% / 38%, ...
AI tells (humanizer catalog): 23 -> 4 (strong 3 -> 0)
```

- **Composite distance**: how far the text is from the author on every reliable feature, in units of
  the author's own variation (allowing for text length). Lower is closer.
- **Author band**: the distances the author's *own* held-out writing gets at a similar length
  (leave-one-out calibration, 10th-90th percentile). Inside the band means indistinguishable from the
  author by these measures. Zero is not the goal; well below the band suggests over-fitting.
- **Sentence rhythm**: spread and shape of sentence lengths, the most visible trait of AI drafts
  (uniform 15-25-word sentences). The line below the table compares the length mix with the author's.
- **Burrows' Delta / Cosine Delta**: classic function-word stylometry.
- **AI tells**: count before -> after; tells the author genuinely uses are excluded.
- **Remaining top gaps**: what still differs, as instructions.
- **Fidelity**: PASS, or every unresolved difference. A failed or missing check is printed at the top.

Humanize report: a table of tells (total, strong, normal, weak) and sentence-length variation for
source and output; a per-pattern before/after table (s01 Not X but Y, s12 Overused AI words, ...);
and the fidelity result.

---

## Command-line tools

Everything the skills do is available as `scripts/idiolect <command>`, in the plugin folder or in
the installed copy under `~/.claude/plugins/cache/idiolect-local/idiolect/<version>/`. A handy alias:

```bash
alias idl=/path/to/idiolect_plugin/idiolect/scripts/idiolect
```

| Task | Command |
|---|---|
| Build or update a voice | `idl profile --voice me --genre paper ~/writing/papers` (add `--add` to grow it) |
| Score any document against a voice | `idl score --voice me --genre paper drafts/intro.md` |
| Re-run calibration | `idl score --voice me --genre paper --calibrate` |
| Print the style brief | `idl brief --voice me --genre paper` |
| Find exemplars for a passage | `idl retrieve --voice me --genre paper --k 3 passage.md` |
| List AI tells in a document | `idl tells drafts/post.md` (`--json` for machine output) |
| Compare numbers, citations and names in two files | `idl fidelity compare original.md rewritten.md` |
| List voices | `idl list` |
| See what extraction keeps and strips | `idl extract ~/writing/papers` (`--text` prints the cleaned text) |
| Create the Python environment | `idl setup` |

The rewrite and humanize loops (`idl rewrite prepare|assemble|finalize`,
`idl humanize prepare|assemble|finalize`, `idl fidelity check|record`) are designed to be driven by
Claude through the skills; each script's `--help` documents them.

---

## Corpus advice

- **Size:** 20k+ words per genre is ideal; 5-20k works with noisier targets; under 5k is weak.
  Frequent function words become reliable around 25-50k words; rare punctuation never fully does.
- **Genre:** profile genres separately. The profiler warns when samples look mixed and names the
  outlier files.
- **Samples:** each file is a sample; files under 1,500 words are grouped. Five or more files give
  the best variance estimates and calibration. With fewer than 3, the corpus is split into
  pseudo-samples.
- **Formats:** .docx and .md give the cleanest text; two-column PDFs are the hardest case.
- **Only the author's writing:** co-authored sections, long quotations and boilerplate dilute the voice.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/idiolect:...` commands don't appear | Run `/reload-plugins`; check `claude plugin list`; if it is missing, redo the install ([Option A](#option-a-install-from-github-recommended) or [Option B](#option-b-install-from-a-local-clone)). |
| `idiolect: Python environment not found` (exit code 3) | Run `bash /path/to/idiolect_plugin/idiolect/setup.sh` (or `idl setup`). |
| `spaCy model en_core_web_sm is missing` | Re-run `setup.sh`; it installs the model. |
| `no profile at ~/.idiolect/voices/...` | Build the voice first with `/idiolect:voice-profile`; check the name and genre with `/idiolect:voice-list`. |
| Warning: corpus is only N words | Add documents with `--add`; targets are noisy under 20k words. |
| Warning: samples may mix genres | Profile the named outlier files under a different `--genre`. |
| A PDF's text looks wrong | Check it with `idl extract file.pdf --text`; prefer a .docx or .md version of the same document. |
| A run reports `incomplete` / paragraph mismatch | Normal mid-run message: a rewritten chunk had a different number of paragraphs. Claude fixes it and re-assembles. |
| The scorecard says FIDELITY FAIL | Read the listed differences; ask Claude to fix them, or edit the output yourself. Do not use the output until it passes. |
| Edits to the plugin source have no effect | See [Updating](#updating). |

---

## Known limitations

- **Short texts:** below ~300 words many features cannot be measured; below ~600 words the score
  separates AI prose from human prose but not one human author from another, and a short AI text can
  already sit inside a wide band. Judge short outputs mainly by the gap list and the rhythm line.
- **Rare features:** features seen fewer than 10 times in the corpus are excluded from targets.
- **Highly technical text:** dense notation and jargon leave little room for voice; the rewrite moves
  less, and the fidelity checker's name heuristic may flag technical terms.
- **Paragraph structure is inherited:** rewrites keep the source's paragraph count (needed for
  paragraph-by-paragraph fidelity checking), so paragraph length is reported but not targeted.
- **Function words are slow to move:** Delta typically improves least. Rewriting changes rhythm,
  punctuation and vocabulary far more readily than the deep habits of closed-class words.
- **The fidelity reviewer is a second opinion, not a proof.** It is noisy on single-word strength
  calls (it has called one synonym too weak and the next too strong). Read the output.
- **Tell detection is pattern matching.** Human writing also contains lists of three, dashes and curly
  quotes (Emerson: ~12 tells per 1k words, mostly those; generic AI prose: ~40). Strong tells are
  rarer in human prose (~1 per 1k vs ~3-4). Real lists of three are kept, so some "forced triad"
  hits always remain. Humanizer's §26 (re-explaining what a reply's reader already knows) is not
  detected.
- English only.

## Appropriate use

This plugin is intended for rewriting text into your own voice, or a voice you have permission to
use. Some journals and institutions require disclosure of AI assistance regardless of how the output
reads.

## Credits

The AI-tell catalog is vendored from [blader/humanizer](https://github.com/blader/humanizer)
(v3.1.0, commit 0ef9fe5, MIT License, Copyright (c) 2025 Siqi Chen), itself based on Wikipedia's
"Signs of AI writing" (WikiProject AI Cleanup). See `THIRD_PARTY_NOTICES.md` and
`skills/idiolect/references/humanizer-patterns.md`.

## Development

```bash
cd /path/to/idiolect_plugin/idiolect
~/.idiolect/venv/bin/python -m pytest                # 77 tests, about 1 minute
~/.idiolect/venv/bin/python -m pytest -m "not slow"  # fast unit tests only
claude plugin validate .
```

The test corpora are public-domain essays from Project Gutenberg (Emerson, *Essays: First Series*;
Chesterton, *All Things Considered*). They validate the stylometry before any rewriting: each
held-out essay is closer to its own author's leave-one-out profile than to the other author's
(20/20).
