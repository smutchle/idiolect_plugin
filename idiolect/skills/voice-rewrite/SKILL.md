---
name: voice-rewrite
description: Rewrite one or more documents (or every document in a folder) in a saved voice, preserving meaning exactly. Use when the user asks to put AI-drafted or other text into their own voice with an existing idiolect profile.
argument-hint: <voice>[/<genre>] <files-or-folders...> [--rounds 3] [--allow-cuts] [--output-dir DIR]
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/scripts/idiolect *), Read, Write, Edit, Glob, Agent
---

# Rewrite documents in a saved voice

Arguments: `$ARGUMENTS`

Parse: first token `<voice>` or `<voice>/<genre>` (genre defaults to `general`; if the voice
has exactly one genre, use it — check with `"${CLAUDE_PLUGIN_ROOT}/scripts/idiolect" list`);
then files and/or folders; optional `--rounds N` (revision rounds, default 3), `--allow-cuts`
(pass it to `prepare`: lets the rewrite cut empty AI puffery, see the idiolect skill) and
`--output-dir DIR`. If the voice does not exist, stop and point to `/idiolect:voice-profile`.

Expand folders to their .md/.txt/.docx/.pdf files, skipping files this plugin produced
(`*.<voice>.*`, `*.scorecard.md`). Confirm the file list with the user if there are more than 5.

On Windows, pass file and folder paths to the script with forward slashes or in double quotes: convert `C:\Users\me\docs` to `C:/Users/me/docs` (Python accepts both; bash mangles unquoted backslashes).

For **each** document, in turn, run the full loop in
`${CLAUDE_PLUGIN_ROOT}/skills/idiolect/SKILL.md` — read that file now and follow it exactly.
Outputs go next to each source as `<stem>.<voice>.<ext>` (pdf -> .md) unless `--output-dir`
is given, in which case pass `--output <dir>/<stem>.<voice>.<ext>` to `prepare`.

When all documents are done, give a one-table summary: document, source -> output composite
distance, inside band?, sentence rhythm inside band?, fidelity result, output path.
