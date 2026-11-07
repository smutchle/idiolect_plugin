"""Shared helpers: paths, spaCy loading, word lists, JSON I/O."""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


# ---------------------------------------------------------------- storage

def idiolect_home() -> Path:
    """User data dir. Lives outside the plugin so voices survive plugin updates."""
    return Path(os.environ.get("IDIOLECT_HOME", Path.home() / ".idiolect")).expanduser()


def voice_dir(name: str, genre: str = "general", create: bool = False) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[A-Za-z0-9_.-]+", genre):
        raise ValueError("voice and genre names may contain only letters, digits, '.', '_' and '-'")
    d = idiolect_home() / "voices" / name / genre
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(*args) -> None:
    """Diagnostics go to stderr so stdout stays machine-readable."""
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------- spaCy

@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError as e:  # pragma: no cover
        raise SystemExit(
            "spaCy model en_core_web_sm is missing. Run the plugin's setup.sh "
            "(or: python -m spacy download en_core_web_sm)."
        ) from e
    nlp.max_length = 3_000_000
    return nlp


def is_word(tok) -> bool:
    """A 'word' for counting purposes: contains a letter, is not punctuation/space."""
    return not tok.is_punct and not tok.is_space and any(c.isalpha() for c in tok.text)


# ---------------------------------------------------------------- word lists

FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}

DISCOURSE_WORDS = {
    "however", "thus", "therefore", "also", "indeed", "moreover", "then", "still",
    "rather", "perhaps", "yet", "so", "furthermore", "nevertheless", "nonetheless",
    "hence", "instead", "meanwhile", "otherwise", "accordingly", "consequently",
    "besides", "likewise", "similarly", "conversely", "namely", "anyway", "though",
    "although", "even", "only", "just", "too", "again", "already", "now", "here",
    "there", "first", "second", "finally", "ultimately", "overall", "additionally",
    "notably", "importantly", "specifically", "essentially", "simply", "actually",
    "really", "quite", "very", "much", "more", "most", "less", "least", "well",
    "not", "no", "never", "always", "often", "sometimes", "usually", "merely",
}

HEDGES = {
    "may", "might", "could", "perhaps", "possibly", "likely", "unlikely", "probably",
    "suggest", "suggests", "suggested", "suggesting", "appear", "appears", "appeared",
    "seem", "seems", "seemed", "somewhat", "apparently", "presumably", "plausibly",
    "tend", "tends", "arguably", "roughly", "approximately", "partly", "potentially",
    "indicate", "indicates", "indicated",
}

BOOSTERS = {
    "clearly", "demonstrate", "demonstrates", "demonstrated", "certainly", "definitely",
    "undoubtedly", "obviously", "surely", "indeed", "always", "never", "prove",
    "proves", "proved", "show", "shows", "showed", "evident", "evidently", "strongly",
    "conclusively", "must", "establish", "establishes", "established", "unquestionably",
}

NOMINAL_SUFFIXES = ("tion", "tions", "sion", "sions", "ment", "ments", "ness",
                    "nesses", "ity", "ities", "ance", "ances", "ence", "ences")

# Suffix heuristic for Latinate vocabulary (documented in feature-guide.md).
LATINATE_SUFFIXES = ("tion", "sion", "ment", "ity", "ance", "ence", "ive", "ous",
                     "ate", "ize", "ise", "ible", "able", "al", "ant", "ent", "ical",
                     "ure", "ory", "ary", "ify")

TRANSITION_OPENERS = {
    "however", "but", "yet", "thus", "therefore", "moreover", "furthermore", "also",
    "still", "nevertheless", "nonetheless", "hence", "so", "and", "or", "instead",
    "meanwhile", "similarly", "likewise", "conversely", "consequently", "accordingly",
    "indeed", "finally", "first", "second", "third", "next", "then", "additionally",
    "now", "again", "besides", "otherwise", "nor", "though", "although", "while",
}
# multiword transitions recognised at paragraph start
TRANSITION_PHRASES = ("in addition", "on the other hand", "in contrast", "by contrast",
                      "as a result", "for example", "for instance", "in particular",
                      "at the same time", "in other words", "that said", "even so")

SUMMARY_OPENERS = ("thus", "therefore", "hence", "so", "in short", "in sum", "in summary",
                   "overall", "ultimately", "in conclusion", "taken together", "in the end",
                   "all of this", "this means", "this suggests", "this shows", "in this way",
                   "together, these", "collectively")

AI_VOCAB = {
    "delve", "delves", "delving", "delved", "tapestry", "intricate", "intricacies",
    "pivotal", "underscore", "underscores", "underscored", "underscoring", "showcase",
    "showcases", "showcasing", "showcased", "navigate", "navigating", "navigates",
    "landscape", "landscapes", "robust", "leverage", "leverages", "leveraging",
    "leveraged", "realm", "realms", "multifaceted", "crucial", "seamless", "seamlessly",
    "testament", "vibrant", "foster", "fosters", "fostering", "holistic", "nuanced",
    "paramount", "embark", "unwavering", "meticulous", "meticulously", "comprehensive",
}

AI_CLOSERS = ("ultimately,", "in conclusion,", "overall,", "in summary,", "in essence,",
              "all in all,", "to sum up,")

CONTRACTION_RE = re.compile(r"^['’](s|re|ve|ll|d|m|t)$|^n['’]t$", re.I)


@lru_cache(maxsize=None)
def zipf(word: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")


@lru_cache(maxsize=None)
def wf_rate_per_1k(word: str) -> float:
    from wordfreq import word_frequency
    return word_frequency(word, "en") * 1000.0


def syllables(word: str) -> int:
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))
