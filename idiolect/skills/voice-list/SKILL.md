---
name: voice-list
description: List saved idiolect voices with their genres, corpus sizes, sample counts and calibration status.
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/scripts/idiolect *)
---

Run `"${CLAUDE_PLUGIN_ROOT}/scripts/idiolect" list` and show the table. Point out any voice/genre
under 20k words (targets will be noisy) and any that is not calibrated (re-run
`"${CLAUDE_PLUGIN_ROOT}/scripts/idiolect" score --voice <v> --genre <g> --calibrate`).
If there are no voices, explain `/idiolect:voice-profile <name> <folder>`.
