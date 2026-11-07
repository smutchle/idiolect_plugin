"""Deterministic detectors for AI-writing tells.

The pattern catalog follows blader/humanizer v3.1.0 (MIT, (c) 2025 Siqi Chen;
https://github.com/blader/humanizer), itself based on Wikipedia's "Signs of AI writing".
Section numbers (s01..s26) match that catalog; skills/idiolect/references/humanizer-patterns.md
has the full guidance. Here we only *find* tells; the model fixes them.

    python tells.py FILE [--json]        report the tells in a document

Tiers (from the catalog): strong = act on one sighting (s01-s05); weak = needs company from
other tells before acting; normal = everything else. In idiolect every pattern is also judged
against the author's own rate (a writing sample overrides the patterns).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------- catalog
# removal: 'wrapper' = no content, always removable; 'cut' = removable only with --allow-cuts
#          (the fidelity reviewer classifies it); 'rephrase' = keep the content, change the form
PATTERNS: dict[str, dict] = {
    "s01": {"name": "Not X but Y", "tier": "strong", "removal": "rephrase",
            "fix": "State the point directly. Keep a contrast only if the negative half corrects a belief the reader "
                   "actually holds, or both halves carry information."},
    "s02": {"name": "One-line closers and dramatic fragments", "tier": "strong", "removal": "cut",
            "fix": "Cut a closer that repeats the point; keep it only if it adds a new fact or consequence. Merge a row "
                   "of fragments into one sentence with a specific claim."},
    "s03": {"name": "Sayings that sound deep", "tier": "strong", "removal": "rephrase",
            "fix": "Replace the aphorism or 'the real question is' dressing with the specific claim."},
    "s04": {"name": "Staged run-up before the point", "tier": "strong", "removal": "wrapper",
            "fix": "Remove the run-up ('Let's dive in', 'Here's the thing') and make the point."},
    "s05": {"name": "Arguing with no one", "tier": "strong", "removal": "cut",
            "fix": "Remove the defense against an objection nobody raised; if it holds a real claim, state the claim."},
    "s06": {"name": "Forced triads", "tier": "normal", "removal": "rephrase",
            "fix": "Keep three items only when the meaning has three parts; otherwise merge or vary the structure. "
                   "Never drop an item the source asserts."},
    "s07": {"name": "Repeated sentence openings", "tier": "normal", "removal": "rephrase",
            "fix": "Merge the sentences, change the subject, or begin with the action."},
    "s08": {"name": "Dashes as the universal connector", "tier": "weak", "removal": "rephrase",
            "fix": "Replace dashes with a period, comma, colon, or parentheses (unless the author uses them: match "
                   "the author's rate)."},
    "s09": {"name": "Stacked qualifiers", "tier": "weak", "removal": "rephrase",
            "fix": "Keep one qualifier where the source supports it; do not change the claim's strength."},
    "s10": {"name": "Hyphenated pairs everywhere", "tier": "weak", "removal": "rephrase",
            "fix": "Drop the hyphen after the noun ('the report is high quality'); keep it before the noun."},
    "s11": {"name": "Passive voice and missing subjects", "tier": "weak", "removal": "rephrase",
            "fix": "Name the actor where it makes the sentence clearer; give verbless fragments a subject."},
    "s12": {"name": "Overused AI words", "tier": "normal", "removal": "rephrase",
            "fix": "Use the plain word (also, important, show, area, strong, careful)."},
    "s13": {"name": "Inflated significance", "tier": "normal", "removal": "cut",
            "fix": "Keep the fact; drop the significance dressing ('pivotal moment', 'evolving landscape', "
                   "'the future looks bright'). End on the last concrete fact."},
    "s14": {"name": "Vague connection or association", "tier": "weak", "removal": "rephrase",
            "fix": "Name the relationship if the source gives it; otherwise keep the wording (never invent a role)."},
    "s15": {"name": "Shallow -ing riders", "tier": "normal", "removal": "cut",
            "fix": "Keep the fact; drop the bolted-on ', highlighting/ensuring/reflecting ...' unless the source "
                   "supports what it claims."},
    "s16": {"name": "Sales language", "tier": "normal", "removal": "rephrase",
            "fix": "State what the thing is, without advertising adjectives."},
    "s17": {"name": "Borrowed authority", "tier": "normal", "removal": "cut",
            "fix": "Use the real source and what it said if the text names it; otherwise cut the unsupported claim."},
    "s18": {"name": "Avoiding is, are, and has", "tier": "normal", "removal": "rephrase",
            "fix": "Use is / are / has instead of serves as, stands as, boasts, features."},
    "s19": {"name": "Bold as decoration", "tier": "normal", "removal": "rephrase",
            "fix": "Remove decorative bold; turn bold-label lists into prose when the labels carry nothing."},
    "s20": {"name": "Decorative headings", "tier": "normal", "removal": "rephrase",
            "fix": "Sentence-case headings; remove emojis, arrows and decorative rules."},
    "s21": {"name": "Curly quotation marks", "tier": "weak", "removal": "rephrase",
            "fix": "Use straight quotes if the target format does (weak alone; many editors auto-curl)."},
    "s22": {"name": "Chatbot residue", "tier": "strong", "removal": "wrapper",
            "fix": "Remove the greeting, praise, offer or sign-off; keep the content it wraps."},
    "s23": {"name": "Knowledge-limit disclaimers and guesses", "tier": "normal", "removal": "cut",
            "fix": "State what the source does not show, or remove the sentence; never fill the gap with a guess."},
    "s24": {"name": "A heading repeated in the first sentence", "tier": "normal", "removal": "cut",
            "fix": "Remove the sentence that restates the heading."},
    "s25": {"name": "Writing about the document instead of its subject", "tier": "weak", "removal": "cut",
            "fix": "Describe the subject, not the document ('the table below compares ...')."},
    # s26 (re-explaining what the reader knows) needs the surrounding conversation; not detectable here.
}
STRONG = {k for k, v in PATTERNS.items() if v["tier"] == "strong"}
# detectable in plain prose (profile-mode text has no markdown or headings)
PROSE_PATTERNS = [p for p in PATTERNS if p not in ("s19", "s20", "s24")]
WEAK = {k for k, v in PATTERNS.items() if v["tier"] == "weak"}


def _ph(*phrases: str) -> re.Pattern:
    return re.compile(r"(?<![\w'])(" + "|".join(phrases) + r")(?![\w])", re.I)


# --------------------------------------------------------------------- sentence-level regexes
S01 = [
    re.compile(r"\b(?:not|n't)\s+(?:just|only|merely|simply)\s+[^.;:!?]{1,80}?[,;—-]?\s+(?:but|it'?s|it is)\b", re.I),
    re.compile(r"\b(?:it'?s|it is|this is|that'?s|this isn'?t|it isn'?t)\s+not\s+(?:about\s+)?[^.;!?]{1,60}?[,;—-]\s*"
               r"(?:it'?s|it is|this is|but)\b", re.I),
    re.compile(r"\bnot\s+(?:a|an|the)\s+\w+(?:\s\w+)?,\s+but\s+(?:a|an|the)\b", re.I),
]
S01_SPLIT_NEG = re.compile(r"\b(?:does not|doesn't|is not|isn't) (?:mean|about)\b", re.I)
S01_SPLIT_POS = re.compile(r"^(?:it|this|that) (?:means|is about|'s about)\b", re.I)
S01_TAIL = re.compile(r",\s+no\s+[a-z]+ing\.?$", re.I)

S02_PHRASES = _ph(r"that is the real win", r"that'?s the real win", r"that distinction matters", r"read that again",
                  r"let that sink in", r"this shows the importance of", r"the message was clear", r"it was a lesson in",
                  r"and that changes everything", r"that'?s the point", r"that is the point", r"full stop")
S02_SPACED = re.compile(r"\b(\w+)\.\s(\w+)\.\s(\w+)\.(?=\s|$)")

S03 = _ph(r"the real question is", r"at its core", r"in reality", r"what really matters", r"fundamentally",
          r"the deeper issue", r"the heart of the matter", r"becomes a trap", r"not a tool but a mirror",
          r"the language of", r"the currency of", r"the architecture of")
S04 = _ph(r"let'?s dive in(?:to)?", r"let'?s (?:explore|break (?:this|it) down|unpack)", r"here'?s what you need to know",
          r"now let'?s look at", r"without further ado", r"heads up", r"quick note", r"here'?s the thing",
          r"the thing is", r"let'?s be honest", r"real talk")
S04_STANDALONE = re.compile(r"^(?:honestly|look|so)\s*[?,!:]", re.I)
S05 = _ph(r"this isn'?t (?:mainly |really |just )?about", r"i'?m not saying", r"to be clear", r"don'?t get me wrong",
          r"this is not to say", r"a tempting approach would be", r"one might be tempted to",
          r"an obvious approach would be", r"you might think", r"it would be easy to just", r"some might say")
# "A, B, and C" - the middle item must not open a clause ("much faster, which reduces X and Y")
_CLAUSE = r"(?!(?:which|that|who|whom|whose|because|when|while|although|though|but|so|as|if|since|where|including|such|like|especially|namely|notably|e\.g|i\.e)\b)"
TRICOLON = re.compile(r"\b[\w-]+(?:\s[\w-]+){0,2},\s" + _CLAUSE + r"[\w-]+(?:\s[\w-]+){0,2},?\s(?:and|or)\s"
                      r"[\w-]+(?:\s[\w-]+){0,2}\b", re.I)
S08_DASH = re.compile(r"—|\s–\s|\s--\s|(?<=\w)\s-\s(?=\w)")
S09 = _ph(r"to be fair", r"it'?s also possible", r"could potentially", r"might arguably", r"may potentially",
          r"could possibly", r"might possibly", r"in some cases it may", r"this is an inference",
          r"it could be argued", r"arguably,? (?:may|might|could)")
S10 = re.compile(r"\b(?:is|are|was|were|be|been|seems?|remains?)\s+(?:very\s+)?(high-quality|well-known|well-documented|"
                 r"long-term|short-term|real-time|client-facing|user-friendly|cutting-edge|state-of-the-art|"
                 r"data-driven|up-to-date|well-designed|well-established)\b", re.I)
S11_FRAG = re.compile(r"^(?:no|zero)\s+\w+(?:\s\w+)?\s+(?:needed|required|necessary)\.?$", re.I)
S12 = _ph(r"actually", r"additionally", r"align(?:s|ed)? with", r"bolster(?:ed|s)?", r"crucial", r"deep dive",
          r"delv(?:e|es|ed|ing)", r"enduring", r"enhanc(?:e|es|ed|ing)", r"garner(?:s|ed)?", r"highlight(?:s|ed|ing)?",
          r"interplay", r"intricate", r"intricacies", r"pivotal", r"quietly", r"robust", r"showcas(?:e|es|ed|ing)",
          r"tapestry", r"testament", r"underscor(?:e|es|ed|ing)", r"valuable", r"vibrant", r"meticulous(?:ly)?",
          r"seamless(?:ly)?", r"multifaceted", r"realm", r"leverag(?:e|es|ed|ing)", r"foster(?:s|ed|ing)?",
          r"holistic", r"paramount", r"embark(?:s|ed)?", r"unwavering", r"navigat(?:e|es|ing) the",
          r"(?:evolving|ever-changing|digital|competitive|technological) landscape")
S13 = _ph(r"stands? as a testament", r"(?:a|the) (?:pivotal|crucial|key|defining|watershed) moment",
          r"plays? (?:a|an) (?:key|crucial|vital|pivotal|significant|important|central|critical) role",
          r"marking (?:a|the)", r"shaping the", r"underscores? (?:its|the) (?:importance|significance)",
          r"reflects? (?:a|the) broader", r"(?:enduring|lasting) legacy", r"setting the stage for",
          r"evolving landscape", r"indelible mark", r"despite (?:these|its|the) challenges", r"continues? to thrive",
          r"the future looks bright", r"exciting times (?:lie )?ahead", r"a step in the right direction",
          r"in today'?s (?:fast-paced|rapidly (?:changing|evolving)|ever-changing|modern|digital) world",
          r"more (?:important|relevant) than ever", r"journey toward(?:s)? excellence")
S14 = _ph(r"(?:is|are|was|were|been) associated with", r"in association with", r"in connection with",
          r"(?:is|are|was|were) (?:linked|tied|connected) to")
S15 = re.compile(r",\s+(highlighting|underscoring|emphasizing|ensuring|reflecting|symbolizing|contributing to|"
                 r"cultivating|fostering|encompassing|showcasing|solidifying|reinforcing|paving the way)\b", re.I)
S16 = _ph(r"rich (?:cultural|history|heritage|tapestry|tradition)", r"profound", r"exemplif(?:y|ies|ied)",
          r"commitment to", r"natural beauty", r"nestled", r"in the heart of", r"groundbreaking", r"renowned",
          r"diverse array", r"breathtaking", r"must-visit", r"stunning", r"world-class", r"state-of-the-art",
          r"unparalleled", r"game-chang(?:er|ing)", r"transformative")
S17 = _ph(r"experts (?:argue|believe|say|agree|suggest|note)", r"observers have (?:cited|noted)", r"industry reports",
          r"some critics", r"several (?:publications|sources)", r"(?:has been|was) (?:featured|profiled) in",
          r"independent coverage", r"active social media presence", r"over [\d,]+ followers",
          r"studies (?:show|have shown)", r"research (?:shows|has shown)")
S18 = _ph(r"serves? as", r"stands? as", r"functions? as", r"operates? as", r"boasts?", r"(?:features|offers) (?:a|an)\b",
          r"represents? (?:a|an) (?:shift|turning point|milestone|new)")
S22 = _ph(r"i hope this helps", r"of course!", r"certainly!", r"great question", r"you'?re absolutely right",
          r"would you like (?:me )?to", r"want me to", r"should i continue", r"let me know if",
          r"here is (?:a|an|the) (?:overview|summary|essay|draft|breakdown|list)", r"feel free to",
          r"happy to help", r"as an ai")
S23 = _ph(r"as of my (?:last|latest) (?:training|knowledge|update)", r"up to my last training",
          r"while specific details (?:are|remain) (?:limited|scarce)", r"based on (?:the )?available information",
          r"not (?:publicly|widely) (?:available|documented|disclosed)", r"in the (?:provided|available) sources",
          r"maintains a low profile", r"it is believed that", r"(?:likely|presumably) (?:grew up|studied|began)")
S25 = _ph(r"the table below", r"as shown below", r"this section (?:is organized|covers|describes|outlines)",
          r"this document (?:describes|covers|outlines)", r"was added to replace", r"compiled from",
          r"generated from", r"anything (?:we|i) could not confirm", r"in this (?:article|post|essay), (?:we|i) will")
CURLY = re.compile(r"[“”]")
BOLD = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
BOLD_LABEL = re.compile(r"^\s*(?:[-*+]|\d+[.)])?\s*\W{0,3}\s*\*\*[^*]+:\*\*|^\s*(?:[-*+]|\d+[.)])?\s*\W{0,3}\s*\*\*[^*]+\*\*:")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿←-⇿⭐✅]")
ALLCAPS_WORD = re.compile(r"(?<![\w-])[A-Z]{3,}(?![\w-])")
STOP_SMALL = {"a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "vs"}


@dataclass
class Hit:
    pattern: str
    text: str
    para: int = -1          # paragraph index (within the scanned list)
    sentence: str = ""

    @property
    def name(self) -> str:
        return PATTERNS[self.pattern]["name"]


def _sentences(text: str) -> list[str]:
    from common import get_nlp
    return [s.text.strip() for s in get_nlp()(text).sents if s.text.strip()]


CLOSER_OPENERS = ("ultimately,", "in conclusion,", "overall,", "in summary,", "in essence,", "all in all,",
                  "to sum up,", "in the end,")


def scan_sentences(sents: list[str], para: int = -1, pos_tags: list[list[str]] | None = None) -> list[Hit]:
    """Sentence-level patterns for one paragraph."""
    hits: list[Hit] = []
    spans: set[tuple[str, int, int, int]] = set()

    def add(pid, m_text, s, start: int | None = None):
        # one hit per pattern per overlapping span (several regexes can match the same text)
        if start is not None:
            si = sents.index(s) if s in sents else -1
            for (p0, s0, a0, b0) in spans:
                if p0 == pid and s0 == si and not (start + len(m_text) <= a0 or start >= b0):
                    return
            spans.add((pid, si, start, start + len(m_text)))
        hits.append(Hit(pid, m_text.strip(), para, s))

    for i, s in enumerate(sents):
        for rx in S01:
            for m in rx.finditer(s):
                add("s01", m.group(0), s, m.start())
        if S01_TAIL.search(s):
            add("s01", S01_TAIL.search(s).group(0), s)
        if i + 1 < len(sents) and S01_SPLIT_NEG.search(s) and S01_SPLIT_POS.search(sents[i + 1]):
            add("s01", s + " " + sents[i + 1], s)
        for m in S02_PHRASES.finditer(s):
            add("s02", m.group(0), s)
        if S02_SPACED.search(s):
            add("s02", S02_SPACED.search(s).group(0), s)
        for m in ALLCAPS_WORD.finditer(s):
            w = m.group(0)
            if w not in ("I", "OK") and not (len(w) <= 5 and any(ch.isdigit() for ch in w)) and \
                    sum(1 for x in ALLCAPS_WORD.finditer(s)) == 1 and len(s.split()) > 3 and w.lower() in (
                    "never", "always", "every", "not", "all", "must", "only", "this", "real", "very", "nothing", "no"):
                add("s02", w, s)
        for rx, pid in ((S03, "s03"), (S04, "s04"), (S05, "s05"), (S09, "s09"), (S12, "s12"), (S13, "s13"),
                        (S14, "s14"), (S16, "s16"), (S17, "s17"), (S18, "s18"), (S22, "s22"), (S23, "s23"),
                        (S25, "s25")):
            for m in rx.finditer(s):
                add(pid, m.group(0), s)
        if S04_STANDALONE.match(s) and len(s.split()) <= 3:
            add("s04", s, s)
        # rhetorical question answered at once ("Is it worth it? Honestly? It depends.")
        if s.rstrip().endswith("?") and i + 1 < len(sents) and not sents[i + 1].endswith("?") \
                and len(sents[i + 1].split()) <= 6:
            add("s04", s + " " + sents[i + 1], s)
        # colon reveal ("The result: a faster pipeline.") is a staged, dramatic line
        if ":" in s:
            pre, _, post = s.partition(":")
            if len(pre.split()) <= 4 and 0 < len(post.split()) <= 10 and "," not in post and not \
                    re.search(r"\d:\d", s):
                add("s02", s, s)
        if TRICOLON.search(s):
            add("s06", TRICOLON.search(s).group(0), s)
        for m in S08_DASH.finditer(s):
            add("s08", m.group(0), s)
        for m in S10.finditer(s):
            add("s10", m.group(0), s)
        if S11_FRAG.match(s):
            add("s11", s, s)
        for m in S15.finditer(s):
            add("s15", m.group(0), s)
        if CURLY.search(s):
            add("s21", "curly quotes", s)
    # stock send-off opener on a paragraph's last sentence
    if len(sents) >= 2 and sents[-1].lower().startswith(CLOSER_OPENERS):
        add("s13", sents[-1].split(",")[0] + ",", sents[-1])
    # s07: three or more consecutive sentences with the same first word
    firsts = [re.sub(r"[^\w']", "", s.split()[0]).lower() if s.split() else "" for s in sents]
    run = 1
    for i in range(1, len(firsts)):
        run = run + 1 if firsts[i] and firsts[i] == firsts[i - 1] else 1
        if run == 3:
            add("s07", " / ".join(sents[i - 2:i + 1])[:200], sents[i])
    # s02: a row of verbless fragments (needs POS)
    if pos_tags:
        frag = 0
        for s, tags in zip(sents, pos_tags):
            is_frag = len(tags) <= 5 and not any(t in ("VERB", "AUX") for t in tags)
            frag = frag + 1 if is_frag else 0
            if frag == 2:
                add("s02", s, s)
    return hits


def is_title_case(h: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][\w'-]*", h)]
    if len(words) < 3:
        return False
    content = [w for w in words[1:] if w.lower() not in STOP_SMALL]
    return bool(content) and sum(w[0].isupper() for w in content) / len(content) >= 0.8 and \
        not all(w.isupper() for w in words)


def _content_words(t: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", t)} - {"this", "that", "with", "from", "have", "what", "your"}


def scan_blocks(blocks, nlp_sents: bool = True) -> list[Hit]:
    """Document-level scan over extract.Block objects (rewrite-mode extraction keeps
    markdown, so formatting tells are visible). Paragraph indices count para/list blocks."""
    from common import get_nlp
    nlp = get_nlp()
    hits: list[Hit] = []
    k = -1
    prev_heading = None
    prev_para_text = None
    paras_seen = 0
    for b in blocks:
        if b.kind == "heading":
            if is_title_case(b.text):
                hits.append(Hit("s20", b.text, -1, b.text))
            if EMOJI.search(b.text) or "→" in b.text:
                hits.append(Hit("s20", b.text, -1, b.text))
            prev_heading = b.text
            continue
        if b.kind not in ("para", "list"):
            continue
        k += 1
        paras_seen += 1
        text = b.text
        doc = nlp(re.sub(r"\*\*|__", "", text))
        sents = [s for s in doc.sents if s.text.strip()]
        stexts = [s.text.strip() for s in sents]
        tags = [[t.pos_ for t in s if not t.is_punct and not t.is_space] for s in sents]
        hits += scan_sentences(stexts, k, tags)
        if BOLD_LABEL.search(text) or len(BOLD.findall(text)) >= 2:
            hits.append(Hit("s19", BOLD.search(text).group(0) if BOLD.search(text) else text[:40], k, text[:120]))
        if b.kind == "list" and (EMOJI.search(text) or "→" in text):
            hits.append(Hit("s20", text[:60], k, text[:120]))
        # s24: first paragraph after a heading is one short sentence restating it
        if prev_heading is not None and len(stexts) == 1 and len(stexts[0].split()) <= 12:
            hw, pw = _content_words(prev_heading), _content_words(stexts[0])
            if hw and len(hw & pw) / len(hw) >= 0.5:
                hits.append(Hit("s24", stexts[0], k, stexts[0]))
        # s02: a one-sentence paragraph that closes a section after a longer one
        if prev_para_text is not None and len(stexts) == 1 and len(stexts[0].split()) <= 10 and \
                len(prev_para_text.split()) >= 30 and b.kind == "para":
            hits.append(Hit("s02", stexts[0], k, stexts[0]))
        prev_heading = None
        prev_para_text = text
    return hits


def summarize(hits: list[Hit], words: int) -> dict:
    by = Counter(h.pattern for h in hits)
    return {"words": words, "total": len(hits),
            "by_pattern": {p: {"name": PATTERNS[p]["name"], "tier": PATTERNS[p]["tier"], "count": n,
                               "per_1k": round(1000 * n / max(words, 1), 2)} for p, n in sorted(by.items())},
            "strong": sum(n for p, n in by.items() if p in STRONG),
            "weak_only": all(p in WEAK for p in by)}


def render(hits: list[Hit], words: int, title: str = "") -> str:
    s = summarize(hits, words)
    L = [f"# AI tells{': ' + title if title else ''}",
         f"{s['total']} tell(s) in {words:,} words; {s['strong']} strong (act on one sighting)."]
    if s["total"] and s["weak_only"]:
        L.append("Only weak tells: these need company from other tells before they justify an edit.")
    by_p: dict[str, list[Hit]] = {}
    for h in hits:
        by_p.setdefault(h.pattern, []).append(h)
    for p in sorted(by_p):
        meta = PATTERNS[p]
        L.append(f"\n## {p} {meta['name']} ({meta['tier']}, {len(by_p[p])})")
        L.append(f"Fix: {meta['fix']}")
        for h in by_p[p][:8]:
            where = f"para {h.para + 1}" if h.para >= 0 else "heading"
            L.append(f"- {where}: \"{h.text[:120]}\"")
        if len(by_p[p]) > 8:
            L.append(f"- ... {len(by_p[p]) - 8} more")
    return "\n".join(L)


def scan_file(path) -> tuple[list[Hit], int]:
    import extract as ex
    doc = ex.extract(path, for_rewrite=True)
    words = sum(len(b.text.split()) for b in doc.blocks if b.kind in ("para", "list"))
    return scan_blocks(doc.blocks), words


def scan_text(text: str) -> tuple[list[Hit], int]:
    import extract as ex
    doc = ex.Document("<text>")
    blocks = ex.read_markdown(text, doc, keep=True)
    words = sum(len(b.text.split()) for b in blocks if b.kind in ("para", "list"))
    return scan_blocks(blocks), words


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    hits, words = scan_file(a.file)
    if a.json:
        print(json.dumps({"summary": summarize(hits, words), "hits": [asdict(h) for h in hits]}, indent=1))
    else:
        print(render(hits, words, Path(a.file).name))


if __name__ == "__main__":
    main()
