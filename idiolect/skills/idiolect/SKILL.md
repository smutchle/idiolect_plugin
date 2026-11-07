---
name: idiolect
description: The idiolect rewrite loop - rewrite a document in a saved author voice using exemplar passages, deterministic style measurement, numeric gap reports, and a fidelity check. Use when rewriting text into a specific author's (usually the user's own) voice with an idiolect profile, or when asked how the idiolect scoring works.
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/scripts/idiolect *), Read, Write, Edit, Agent
---

# The idiolect rewrite loop

**Core principle:** you are good at imitating a voice from examples and bad at counting.
Python measures; you write. Never estimate a statistic yourself (sentence lengths, rates,
"this sounds closer now"). Always run the scorer and act on its numbers.

`IDL` below means `${CLAUDE_PLUGIN_ROOT}/scripts/idiolect`.
Feature meanings and how to act on each gap: `${CLAUDE_PLUGIN_ROOT}/skills/idiolect/references/feature-guide.md`.
AI tells: the humanizer catalog (`references/humanizer-patterns.md`, vendored from blader/humanizer
v3.1.0, MIT - read its first section, "How idiolect applies this catalog") and the detector/scoring
map in `references/ai-tells.md`. Tells are found by `tells.py`; prompts list them; you fix them.

## 1. Prepare

```
IDL rewrite prepare --voice <v> --genre <g> --rounds <n> [--allow-cuts] [--output <path>] <source-file>
```

`--allow-cuts` (off by default: fidelity first) lets the rewrite remove empty content that the
humanizer catalog says to cut - restating closers, significance and sales dressing, send-off
paragraphs, unsupported unnamed authority, knowledge-limit disclaimers. Use it for AI drafts heavy
with puffery; leave it off for drafts whose every sentence the user wants kept.

This extracts the source (references, captions, equations, code, tables become protected
blocks that are passed through verbatim), splits it into 150-400-word chunks without splitting
paragraphs, writes the style brief (`brief.md`), retrieves 4 exemplars per chunk, writes one
self-contained prompt per chunk to `<run>/prompts/r1/<chunk>.md`, and scores the untouched
source (round 0). Note the `run_dir` it prints and the source's distance vs. the band.

## 2. Rewrite every chunk (round 1)

For each `<run>/prompts/r1/<chunk>.md`: read it and write the rewritten passage to the path
it names (`<run>/rewrites/<chunk>.md`), and nothing else in that file.

- Up to ~8 chunks: do them yourself, one at a time, reading each prompt fresh.
- More than ~8 chunks: you may delegate batches of 4-6 prompt files to `general-purpose`
  subagents ("Read each of these prompt files and do exactly what it says: <paths>"). Each
  prompt file is self-contained.

While writing:
- Imitate the exemplars' *moves*: how sentences open, where clauses attach, which connectives
  carry the logic, how long and short sentences are mixed. Do not copy their content.
- Follow the rhythm target line: it tells you roughly how many short, medium and long sentences
  the author would use for a passage of this length. Short sentences must be genuinely short.
- Fix every tell listed under "AI tells found in this passage" (quoted, with the fix for each). Tells
  marked as author habits stay at the author's rate. A paragraph that is pure chatbot residue (or,
  with `--allow-cuts`, empty content) is written as `[CUT]`; it still counts as a paragraph.
- Keep the paragraph count exactly; keep citations, numbers, markdown and inline code verbatim.
- Never raise or lower the strength of a claim. Never add an example, caveat or opinion.

## 3. Assemble, score, revise

```
IDL rewrite assemble <run>
```

This assembles the **whole document**, scores it globally (statistics are never judged per
chunk), records the round, tracks the best-scoring version, and applies the stopping rules.
Its JSON `status` is one of:

- `incomplete` - some rewrites are missing or changed the paragraph count. Fix those files and
  run assemble again (this does not consume a round).
- `revise` - the top 3-5 gaps were attributed to the chunks that contribute most to each.
  Revision prompts are in `<run>/prompts/r<k>/`. For each, overwrite `<run>/rewrites/<chunk>.md`
  as instructed. Change only what the gaps require; leave chunks without a prompt untouched.
  Revisions always start from the best version so far (assemble restores it if the last round
  got worse). Then run assemble again.
- `stop` - the loop is over. Reasons: the composite distance and sentence rhythm are inside the
  within-author band; or `--rounds` revision rounds are used up; or a round improved distance by
  less than 5%. The best-scoring version (not the latest) has been restored into `rewrites/`.

Inside the band is the goal; distance zero is not (below the band's low end means
over-fitting, e.g. stuffing signature phrases).

## 4. Fidelity check (mandatory)

```
IDL fidelity check <run>
```

Deterministic part: every number, percentage, unit, citation key and capitalized named entity in
the source must appear in the output and vice versa. It also writes paragraph-aligned
source/output pairs to `<run>/fidelity/pairs/` and prints the list of pair files.

Model part - a **separate, narrow pass with a fresh context**. Spawn one `general-purpose`
subagent per batch of up to 6 pair files (in parallel when there are several) with exactly this
prompt, filling in the paths:

> Compare each SOURCE paragraph with its REWRITE in these files: <paths>. List differences in
> meaning only - ignore wording, style, rhythm, formatting (bold, emojis, list labels) and
> paragraph-internal reordering. Report any claim that was added, dropped, or changed in strength
> (e.g. "may" -> "will", "suggests" -> "shows", "some" -> "most"), and any changed number, entity,
> citation, scope or causal direction. Removing chatbot residue (greetings, "I hope this helps",
> "Let me know if...") or a staged run-up ("Let's dive in") is intended: report each such removal
> once with type "allowed_cut". [ALLOW-CUTS SENTENCE] Reply with only a JSON array; one object per
> difference: {"chunk": "<chunk id from the file name>", "paragraph": <n>,
> "type": "added|dropped|strength|changed|allowed_cut", "source": "<short quote>",
> "rewrite": "<short quote>", "explanation": "<one sentence>"}. Reply [] if there are none.

Replace `[ALLOW-CUTS SENTENCE]` with nothing for a normal run. For a run prepared with
`--allow-cuts`, replace it with:

> Removing empty content is also permitted in this run: significance or sales dressing (e.g.
> "pivotal", "groundbreaking", "enduring", "transformative", "the future looks bright"), a closing
> line that only restates, a bolted-on ", highlighting/reflecting/contributing to ..." phrase,
> synonym-cycling restatements, unsupported claims of significance, knowledge-limit disclaimers, and
> send-off paragraphs. Report such a removal as "allowed_cut" only if the removed text carried no
> fact, number, name or checkable claim; otherwise report it as "dropped".

`allowed_cut` findings are listed in the scorecard but do not fail the check. Expect the reviewer to
be noisy on single-word strength calls (it may call one synonym too weak and the next too strong);
fix what is a real change, and say which flags you judged false positives.

Save the merged arrays to `<run>/fidelity/model.json`, then:

```
IDL fidelity record <run> <run>/fidelity/model.json
```

If anything is flagged (deterministic or model): fix the flagged passages in
`<run>/rewrites/<chunk>.md` (restore the source's meaning; keep the voice), then re-run
`fidelity check` and the model pass on the changed chunks only, and `record` again. Do not
dismiss a flag without fixing it unless it is clearly a false positive; say which in the summary.

## 5. Finalize

```
IDL rewrite finalize <run>
```

The scorecard also shows AI tells before -> after (humanizer catalog, author habits excluded).
Writes the output in the source's format (md/txt; .docx is edited in place so styles, tables and
images survive; .pdf becomes .md), re-scores the final text, and writes the scorecard next to the
output. Show the user the scorecard: source vs. output distance, the within-author band, the
sentence-rhythm result, the remaining top gaps, and the fidelity result. **Unresolved fidelity
issues must be stated plainly at the top of your reply.**

## Prompting rules (for you and any subagent)

- Never paste raw profile JSON into a rewrite; the brief, exemplars and gap instructions are the
  interface.
- Exemplars are style only: never import their topic words, facts, names or examples.
- Suppress the AI tells in `references/ai-tells.md` unless the brief says the author uses them.
- Do not chase every gap. Fix the 3-5 the assembler targets per round; numbers are whole-document
  targets, not per-sentence quotas.
- If a gap cannot be fixed without changing meaning (e.g. "use more 'I'" in a passage with no
  stance), leave it and mention it.

## Appropriate use

This is for rewriting text into the user's own voice, or a voice they have permission to use.
Some journals and institutions require disclosure of AI assistance regardless of how the output
reads; if the document is headed for one, remind the user once.
