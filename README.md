# idiolect_plugin

A Claude Code plugin marketplace containing **[idiolect](idiolect/README.md)**: learn a writing voice
from a folder of documents, rewrite documents in that voice without changing what they say, and
remove AI-writing tells (the [blader/humanizer](https://github.com/blader/humanizer) catalog).

## Install

```bash
# 1. add this marketplace and install the plugin
claude plugin marketplace add smutchle/idiolect_plugin
claude plugin install idiolect@idiolect-local --scope user

# 2. one-time Python environment (~/.idiolect/venv; needs network once)
bash ~/.claude/plugins/cache/idiolect-local/idiolect/*/setup.sh
```

Then type `/reload-plugins` in an open Claude Code session (new sessions load it automatically) and
try `/idiolect:voice-list`.

```text
/idiolect:voice-profile me ~/writing/papers --genre paper     # build your voice once
/idiolect:voice-rewrite me/paper drafts/intro_ai.md           # rewrite a draft in it
/idiolect:humanize drafts/post.md                             # remove AI tells, no voice needed
```

Full documentation, all options and examples: **[idiolect/README.md](idiolect/README.md)**.

License: MIT (see [idiolect/LICENSE](idiolect/LICENSE)). The AI-tell catalog is from
blader/humanizer (MIT); see [idiolect/THIRD_PARTY_NOTICES.md](idiolect/THIRD_PARTY_NOTICES.md).
