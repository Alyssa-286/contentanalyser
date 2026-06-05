"""
Text cleaning utilities for removing noise, HTML, URLs, citation markers,
and formatting inconsistencies from raw text before and after summarization.
"""

import re

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

URL_PATTERN: re.Pattern = re.compile(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}'
    r'\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
)

HTML_PATTERN: re.Pattern = re.compile(r'<[^>]+>')

WHITESPACE_PATTERN: re.Pattern = re.compile(r'\s+')

# Wikipedia / academic citation patterns e.g. [1], [22], [22]: 488, [citation needed],
# [verification needed], [clarification needed], [a], [note 1], [nb 1]
CITATION_BRACKET: re.Pattern = re.compile(
    r'\[\s*(?:\d+(?:[,\s]*\d+)*(?::\s*[\d\w\s,–\-]+)?'
    r'|citation needed|verify|verification needed|clarification needed'
    r'|dubious|discuss|failed verification|better\s+source|note\s*\d*'
    r'|nb\s*\d*|when\?|who\?|where\?|which\?|why\?|how\?|page needed'
    r'|[a-z])\s*\]',
    re.IGNORECASE
)

# Trailing edit/section markers that leak from Wikipedia parsing
WIKI_EDIT_MARKER: re.Pattern = re.compile(
    r'\[edit\]|\[update\]|\[show\]|\[hide\]|\[expand\]|\[collapse\]',
    re.IGNORECASE
)

# Parenthetical "see also" / cross-reference clutter  — "(see X)" "(compare Y)"
SEE_ALSO_PATTERN: re.Pattern = re.compile(
    r'\(\s*(?:see|cf\.|compare|also|viz\.?)\s+[^)]{1,80}\)',
    re.IGNORECASE
)

# Standalone number-only sentences or very-short artifact sentences
# (used as a filter, not a replacement)
NOISE_SENTENCE: re.Pattern = re.compile(
    r'^\s*(\d+\.?\s*){1,5}\s*$'  # e.g. "1. 2. 3."
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def strip_urls(text: str) -> str:
    """Remove all HTTP/HTTPS URLs from the input text."""
    if not text:
        return ""
    return URL_PATTERN.sub("", text)


def strip_html(text: str) -> str:
    """Remove HTML tags from the input text."""
    if not text:
        return ""
    return HTML_PATTERN.sub("", text)


def strip_citations(text: str) -> str:
    """
    Remove inline citation markers and edit-section links that commonly appear
    in text scraped from Wikipedia and academic sources.

    Examples removed:
        [1]  [22]  [22]: 488  [1,2,3]  [citation needed]
        [verification needed]  [clarification needed]  [edit]  [a]
    """
    if not text:
        return ""
    text = CITATION_BRACKET.sub("", text)
    text = WIKI_EDIT_MARKER.sub("", text)
    return text


def clean_sentence(sentence: str) -> str:
    """
    Clean a single extracted sentence for display:
    - Strip citations, HTML, URLs
    - Remove dangling punctuation from stripping
    - Normalize whitespace
    - Ensure the sentence ends with proper punctuation
    """
    s = strip_html(sentence)
    s = strip_urls(s)
    s = strip_citations(s)
    s = SEE_ALSO_PATTERN.sub("", s)
    # Collapse whitespace
    s = WHITESPACE_PATTERN.sub(" ", s).strip()
    # Remove leading punctuation artifacts (e.g. ", and the...")
    s = re.sub(r'^[\s,;:\-–—]+', '', s).strip()
    # Remove trailing incomplete parentheses / brackets
    s = re.sub(r'[\(\[\{][^)\]\}]{0,40}$', '', s).strip()
    # Ensure sentence ends with a period if it doesn't already end with punctuation
    if s and s[-1] not in '.!?':
        s += '.'
    return s


def is_noise_sentence(sentence: str, min_words: int = 6) -> bool:
    """
    Return True if a sentence is too short, numeric-only, or otherwise
    not worth including in a summary.
    """
    s = sentence.strip()
    if not s:
        return True
    if NOISE_SENTENCE.match(s):
        return True
    words = s.split()
    if len(words) < min_words:
        return True
    # Reject sentences that are mostly non-alpha (e.g. table rows, code snippets)
    alpha_chars = sum(1 for c in s if c.isalpha())
    if alpha_chars / max(len(s), 1) < 0.4:
        return True
    return False


def clean_text(text: str) -> str:
    """
    Full pipeline clean for raw input text before summarization:
    Strip HTML → URLs → citations → normalize whitespace.
    Preserves sentence boundaries (does NOT collapse paragraphs to single line).
    """
    if not text:
        return ""

    cleaned = strip_html(text)
    cleaned = strip_urls(cleaned)
    cleaned = strip_citations(cleaned)
    # Normalize runs of whitespace but preserve single newlines (sentence boundaries)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)          # collapse spaces/tabs
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)        # max 2 consecutive newlines
    return cleaned.strip()
