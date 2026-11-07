---
name: voice-profile
description: Build or update a reusable voice (stylometric profile) from a folder or list of the author's documents (.pdf, .docx, .md, .txt). Use when the user wants to capture their own writing voice, or a voice they have permission to use, so documents can later be rewritten in it.
argument-hint: <voice-name> <folder-or-files...> [--genre paper] [--add]
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/scripts/idiolect *), Read
---

# Build a voice profile

Arguments: `$ARGUMENTS`

Parse them as: first token = voice name (letters, digits, `.`, `_`, `-`); then one or more
files or directories; optional `--genre <name>` (default `general`; use e.g. `paper`, `blog`,
`email`, `grant` — profiles are keyed by voice/genre, and mixing genres in one profile blurs
it); optional `--add` (merge these documents into the voice's existing sources and rebuild).
If the voice name or paths are missing, ask for them.

On Windows, pass file and folder paths to the script with forward slashes or in double quotes: convert `C:\Users\me\docs` to `C:/Users/me/docs` (Python accepts both; bash mangles unquoted backslashes).

## Steps

1. Run (directories are searched recursively for .pdf/.docx/.md/.txt):

   ```
   "${CLAUDE_PLUGIN_ROOT}/scripts/idiolect" profile --voice <name> --genre <genre> [--add] <paths...>
   ```

   If it exits with code 3 the Python environment is missing: run
   `"${CLAUDE_PLUGIN_ROOT}/scripts/idiolect" setup` once (it creates `~/.idiolect/venv` and
   downloads the spaCy model; this needs network access), then retry.

   The command extracts and cleans the text (stripping references, captions, equations,
   affiliations, acknowledgments, headers/footers — the log on stderr says what was stripped
   per file), measures ~300 style features, builds the exemplar bank, and runs leave-one-out
   calibration. It takes roughly 10 seconds per 70k words.

2. Report to the user, in plain language and briefly:
   - corpus size (words, files, samples) and whether it meets the recommended >= 20k words;
   - the author's 5-8 most distinctive habits (from the "Most distinctive habits" section —
     restate them as plain English, e.g. "uses semicolons about 14x as often as generic prose");
   - the sentence-rhythm line (median, spread, short/long shares) — users care about this most;
   - the calibration band at ~1000 words and where generic AI-style text lands relative to it;
   - every warning, verbatim in substance: low-reliability features, genre mixing (suggest
     splitting into separate `--genre` profiles), short corpus.
3. Tell the user where the voice is stored (`~/.idiolect/voices/<name>/<genre>/`, persists
   across plugin updates) and how to use it: `/idiolect:voice-rewrite <name> <file-or-folder>`.
   To add documents later: `/idiolect:voice-profile <name> <new-files> --genre <genre> --add`.

Do not paste the raw profile JSON.
