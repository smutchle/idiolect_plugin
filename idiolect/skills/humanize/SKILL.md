---
name: humanize
description: Remove AI-writing tells from documents (the humanizer catalog - not-X-but-Y contrasts, one-line closers, staged openers, forced triads, dashes, inflated claims, sales language, stock AI words, bold labels, chatbot residue) without changing what they say. No voice profile needed; with one, use /idiolect:voice-rewrite instead, which removes tells as part of matching the voice.
argument-hint: <files-or-folders...> [--strict] [--rounds 2] [--output-dir DIR] [--report-only]
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/scripts/idiolect *), Read, Write, Edit, Glob, Agent
---

# Humanize documents (no voice profile)

Arguments: `$ARGUMENTS`

`IDL` = `${CLAUDE_PLUGIN_ROOT}/scripts/idiolect`. Pattern catalog and fixes:
`${CLAUDE_PLUGIN_ROOT}/skills/idiolect/references/humanizer-patterns.md` (read its first section,
"How idiolect applies this catalog"). Detector/scoring summary: `references/ai-tells.md`.

Parse: files and/or folders (expand folders to .md/.txt/.docx/.pdf, skipping `*.humanized.*` and
`*.report.md`); `--strict` (fidelity first: reword instead of cut, see below); `--rounds N` (default 2);
`--output-dir DIR`; `--report-only` (just list the tells: run `IDL tells <file>` for each and stop).

If the user has a voice profile for this kind of text, suggest `/idiolect:voice-rewrite` instead:
it removes the same tells, treats tells the author really uses as habits, and matches the voice.

For **each** document:

1. **Prepare.** `IDL humanize prepare <file> [--strict] --rounds <n> [--output <dir>/<stem>.humanized.<ext>]`
   It scans the document, prints the tell report, writes one prompt per chunk that has tells or
   machine-even sentence lengths to `<run>/prompts/r1/`, and copies clean chunks unchanged.
2. **Rewrite.** For each prompt file, read it and write the passage to the path it names - nothing else
   in that file. Follow the prompt's process (fix every listed tell, self-audit for surviving tells,
   check you added and dropped nothing). Keep the paragraph count; a paragraph that is pure chatbot
   residue is written `[CUT]`. More than ~8 prompts: delegate batches of 4-6 to `general-purpose`
   subagents ("Read each of these prompt files and do exactly what it says: <paths>").
3. **Assemble.** `IDL humanize assemble <run>` re-scans the whole document. `revise` -> do the prompts
   in `<run>/prompts/r<k>/` and assemble again; `incomplete` -> fix the named files; `stop` -> go on.
4. **Fidelity (mandatory).** `IDL fidelity check <run>`, then the model pass exactly as in
   `${CLAUDE_PLUGIN_ROOT}/skills/idiolect/SKILL.md` step 4 (fresh `general-purpose` subagent, narrow
   prompt, including the `allowed_cut` sentence and - unless the run used `--strict` - the
   allow-cuts sentence, since cuts are on by default here), then
   `IDL fidelity record <run> <run>/fidelity/model.json`. Fix every flagged passage in
   `<run>/rewrites/`, re-check the changed chunks, record again. Plain wording must keep the claim's
   strength ("pivotal" -> "decisive", not "important").
5. **Finalize.** `IDL humanize finalize <run>` writes the output (same format; .docx edited in place;
   .pdf -> .md) and `<stem>.humanized.report.md`: tells before/after by pattern, sentence-length
   variation, fidelity. Show the report; put any unresolved fidelity issue at the top of your reply.

## Cuts (default) vs. --strict
- Default for humanize: humanizer's cutting rules apply - chatbot residue, staged run-ups, restating
  closers, significance and sales dressing, send-off paragraphs, -ing riders, synonym cycling,
  unsupported unnamed authority, knowledge-limit disclaimers, heading echoes and
  writing-about-the-document may be removed when they carry no fact, number, name or checkable
  claim (a whole paragraph as `[CUT]`). The fidelity reviewer reports each removal as `allowed_cut`
  (listed, not a failure) or `dropped` (a failure). Facts, numbers, names and list items the source
  asserts are always kept.
- `--strict`: fidelity first, as in voice rewrites - only chatbot residue and staged run-ups are
  removed; every other tell is **rephrased** with every claim kept at its strength. Puffery-heavy
  text then keeps its (reworded) puffery, because the fidelity check treats it as claims.

Never add opinions, reactions, anecdotes, facts, names, numbers or quotes that are not in the source.
