# Feature guide

All rates are per 1,000 words unless noted. Every feature in a profile stores the pooled value,
per-sample values, a between-sample spread, and a reliability flag (high >= 30 occurrences,
medium >= 10, low < 10 -> excluded from targets).

## How a gap is judged
`z = (text value - author value) / sd_eff`, where `sd_eff` combines the author's between-sample
spread (robust: a single unusual sample cannot inflate it) with the sampling noise expected at the
text's length. Short texts are therefore not punished for noise, and features that cannot be
estimated at a text's length are listed as "not measurable". Scale-type statistics (sentence-length
mean/SD/percentiles, paragraph lengths) are compared on a log scale because they vary
multiplicatively. AI tells are one-sided. The **composite** is the weighted mean, over feature
groups, of the mean |z| within each group (z capped at 6). Group weights: rhythm 2, function words
2, length 1, punctuation 1, syntax 1, AI tells 1, lexical 0.75, phrases 0.75, structure 0.5,
openers 0.5, sentence-length histogram 0.5.

The **within-author band** comes from leave-one-out calibration: each sample, whole and in chunks
of ~250-4000 words, is scored against a profile built from the other samples. The band is the
10th-90th percentile of those distances at the nearest length. "Inside" means at or below the 90th
percentile.

## Groups and how to act on them

### Rhythm (highest weight) - sentence length spread and shape
- `sent_len_w1`: distance between the text's sentence-length distribution and the author's, on
  median-centered log lengths (so it measures *shape*: spread and tails, not average length).
  Judged against how far the author's own passages of the same length typically wander. The gap
  report names the length ranges that are off ("35+ words: 0% vs the author's 14%"). Fix by adding,
  merging or splitting sentences in the targeted passages; revision prompts list each passage's
  current sentence lengths so you don't have to count.
- `sent_len_sd`, `sent_loglen_sd`: spread. AI drafts are typically half the author's spread.
- `sent_short_share` (<10 words), `sent_long_share` (>35 words).
- `sent_len_autocorr`: negative = the author alternates long and short sentences.

### Length
`sent_len_mean`, percentiles p10-p90, paragraph lengths. Paragraph count is fixed by the source,
so paragraph-length gaps are reported but not turned into instructions.

### Function words (`fw:<word>`) - the core idiolect signal
Rates of the author's ~150 most frequent closed-class words (determiners, prepositions, pronouns,
conjunctions, auxiliaries, particles) plus a fixed list of discourse adverbs and connectives, which
is always tracked so that *avoidance* (e.g. never "moreover") is part of the profile. Burrows' Delta
and Cosine Delta are computed over these. Fix overuse by substituting the author's preferred word
(the gap names it: "'however' ... the author prefers 'but' (4.2/1k)"); fix underuse by letting the
word do work it naturally does (e.g. "which" for relative clauses, "but" for contrast).

### Punctuation
Semicolons, colons, dashes, parentheses, questions, exclamations, quotation marks per 1k; commas per
sentence. Quotation marks are scored but never turned into an instruction (adding them would invent
quotations).

### Syntax and stance
Passive share of clauses (spaCy `nsubjpass`/`csubjpass`), nominalizations (-tion/-ment/-ness/-ity/
-ance/-ence nouns of 7+ letters), first-person singular/plural and second-person rates, hedges,
boosters, contractions, mean dependency depth, subordinate clauses per sentence, sentence-opener POS
shares and the top-20 opening words. Pronoun and hedge gaps must never be fixed by inventing stance,
opinions or experiences, or by changing claim strength.

### Lexical texture
MATTR (moving-average type-token ratio, window 500), mean word length, share of 3+-syllable words,
and a Latinate proxy: share of words of 6+ letters ending in a Latinate suffix (-tion, -sion, -ment,
-ity, -ance, -ence, -ive, -ous, -ate, -ize/-ise, -ible/-able, -al, -ant/-ent, -ical, -ure, -ory/-ary,
-ify). The proxy over-counts some Germanic words ("total") and misses unsuffixed Latinate words; it is
a texture signal, not etymology.

### Structure
Paragraphs opening with a transition, paragraphs closing with a summary sentence, list items,
citation style (numeric `[3]` vs author-year `(Smith, 2020)`) and citation rate. Citations are never
targets.

### Signature phrases (`phrase:<ngram>`)
Up to 30 distinctive 2-4-grams built only from function words and general vocabulary (wordfreq Zipf
>= 4.5; nouns only if very common, Zipf >= 5.2), never topic words. They must occur in at least 40%
of samples. Ranked by per-token log ratio of observed frequency against general-English frequency
(wordfreq), weighted by log count. Use them sparingly; overuse is a classic over-fitting failure.

### AI tells (`ai_*`)
See `ai-tells.md`. One-sided.

## Reliability and genre
- Rare features (fewer than 10 occurrences in the corpus) are excluded from targets and listed as
  warnings. With 25-50k words, frequent function words are reliable; exclamation marks usually are
  not.
- Samples: each file is a sample; files under 1,500 words are grouped; fewer than 3 samples ->
  the corpus is split into pseudo-samples so variance can still be estimated.
- Genre mixing is flagged when at least 3 major features vary between samples by more than 4x the
  expected sampling noise. Profile different genres separately (`--genre`).
