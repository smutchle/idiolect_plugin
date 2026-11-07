# AI tells: what is detected and how it is scored

The catalog is humanizer v3.1.0 (`humanizer-patterns.md`, vendored, MIT). This page maps it onto
idiolect's detectors (`scripts/tells.py`) and scorer.

## Detection
`tells.py` scans sentences and paragraphs and reports each hit with its quote and paragraph number.
Run it on any file: `IDL tells <file>`.

| id | Pattern | Tier | Default handling | Detector (summary) |
|---|---|---|---|---|
| s01 | Not X but Y | strong | rephrase | not just/only/merely X, but Y; it's not X, it's Y; split "This does not mean X. It means Y."; clipped ", no guessing" tail |
| s02 | One-line closers, dramatic fragments | strong | rephrase (cut when cuts are on) | stock closers ("Let that sink in"); one-sentence paragraph after a long one; row of verbless fragments; colon reveals; one emphatic ALL-CAPS word; "every. single. day." |
| s03 | Sayings that sound deep | strong | rephrase | "the real question is", "at its core", "the language of", ... |
| s04 | Staged run-up | strong | **remove** | "Let's dive in", "Here's the thing", standalone "Honestly?"; question answered at once |
| s05 | Arguing with no one | strong | rephrase (cut when cuts are on) | "To be clear", "I'm not saying", "A tempting approach would be", ... |
| s06 | Forced triads | normal | rephrase, never drop an item | "A, B, and C" with short items (not "A, which B and C") |
| s07 | Repeated sentence openings | normal | rephrase | 3+ consecutive sentences with the same first word |
| s08 | Dashes everywhere | weak | rephrase | em/en dashes, " -- ", spaced hyphen |
| s09 | Stacked qualifiers | weak | rephrase, keep claim strength | "could potentially", "might arguably", ... |
| s10 | Hyphenated pairs after the noun | weak | rephrase | "is high-quality", "remains long-term" |
| s11 | Missing subjects | weak | rephrase | "No configuration needed." |
| s12 | Overused AI words | normal | rephrase | delve, crucial, pivotal, robust, landscape, showcase, underscore, ... |
| s13 | Inflated significance and send-offs | normal | rephrase (cut when cuts are on) | "plays a crucial role", "the future looks bright", "In today's fast-paced world", closing "Ultimately,/In conclusion," |
| s14 | Vague association | weak | rephrase only if the source names the relation | "is associated with", "linked to" |
| s15 | Shallow -ing riders | normal | rephrase (cut when cuts are on) | ", highlighting/ensuring/reflecting ..." |
| s16 | Sales language | normal | rephrase | nestled, groundbreaking, stunning, transformative, ... |
| s17 | Borrowed authority | normal | rephrase (cut when cuts are on) | "experts argue", "has been featured in", "studies show" |
| s18 | Avoiding is/are/has | normal | rephrase | serves as, stands as, boasts, features a |
| s19 | Bold as decoration | normal | rephrase | bold labels ("**Speed:**"), repeated bold |
| s20 | Decorative headings | normal | rephrase | Title Case headings, emojis, arrows |
| s21 | Curly quotes | weak | leave unless the format needs straight quotes | “ ” |
| s22 | Chatbot residue | strong | **remove** | "Great question!", "I hope this helps", "Let me know if" |
| s23 | Knowledge-limit disclaimers | normal | rephrase (cut when cuts are on) | "While specific details are limited", "not publicly available" |
| s24 | Heading repeated in first sentence | normal | rephrase (cut when cuts are on) | one short sentence after a heading sharing most of its words |
| s25 | Writing about the document | weak | rephrase (cut when cuts are on) | "the table below", "this section covers" |
| s26 | Re-explaining what the reader knows | - | not detected | needs the conversation around a reply |

"Cuts are on" means `voice-rewrite --allow-cuts`, or `humanize` without `--strict` (cuts are the
humanize default). "Remove" patterns are wrappers with no content: always removed; a paragraph that is nothing but a
wrapper is written `[CUT]`. Weak tells need company (3+ tells in the passage) before they are listed.

## Scoring
- Prose patterns (all but s19, s20, s24, which need markdown/headings) are profile features
  `tell:sNN` (rate per 1,000 words) in the `ai_tells` group.
- **One-sided:** only an excess over the author's own rate counts. A tell the author uses (>= 0.5/1k,
  seen >= 10 times) is an *author habit*: prompts say "keep about N here" instead of "remove".
- **Strong tells count on one sighting** (the usual minimum of two expected events is waived).
- Scorecards (voice rewrite) and reports (humanize) show tells before -> after, by pattern.

## Never fix a tell by changing meaning
Removing "not just X but Y" must not drop X if the source asserts X. Breaking a triad must keep all
the items the source asserts. Plain wording must keep the claim's strength ("pivotal" -> "decisive",
not "important"; "showcases" -> "shows", not "suggests"). Fidelity outranks style.
