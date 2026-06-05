"""
TextRank extractive summarizer using TF-IDF representation and NetworkX PageRank.

Key improvements over naive TextRank:
- Pre-filters noise sentences before scoring
- Post-cleans each selected sentence (strips citations, normalises punctuation)
- Caps summary length in WORDS (not just sentence count) to avoid wall-of-text
- Groups sentences into readable paragraphs
- Has a hard MAX_WORDS cap so long Wikipedia articles still give concise output
"""

from summarizer.preprocessor import TextPreprocessor
from typing import List, Dict, Optional, Any, Tuple
import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_cleaner import clean_sentence, is_noise_sentence


# Maximum words in the final summary regardless of ratio
MAX_SUMMARY_WORDS = 350
# Minimum sentence length (words) to be considered for selection
MIN_SENTENCE_WORDS = 8
# Target sentences per paragraph group in the output
SENTENCES_PER_PARAGRAPH = 3


class ExtractiveSummarizer:
    """
    ExtractiveSummarizer extracts key sentences from a text document using
    the TextRank algorithm based on TF-IDF cosine similarities and PageRank.

    Output is clean, readable prose with:
    - No citation markers ([1], [citation needed], etc.)
    - Proper paragraph breaks every few sentences
    - Hard word-count cap to prevent wall-of-text summaries
    """

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_and_filter(self, sentences: List[str]) -> Tuple[List[str], List[int]]:
        """
        Clean each sentence and remove noise. Returns:
          - cleaned sentences list
          - mapping from cleaned index → original index
        """
        cleaned, index_map = [], []
        for orig_idx, sent in enumerate(sentences):
            c = clean_sentence(sent)
            if not is_noise_sentence(c, min_words=MIN_SENTENCE_WORDS):
                cleaned.append(c)
                index_map.append(orig_idx)
        return cleaned, index_map

    def _get_sentence_scores(self, sentences: List[str]) -> List[float]:
        """
        Compute TextRank/PageRank scores for a list of sentences.
        """
        n = len(sentences)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # Build TF-IDF matrix
        try:
            vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return [1.0 / n] * n

        # Pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Build weighted graph
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                score = similarity_matrix[i, j]
                if score > 0.1:
                    graph.add_edge(i, j, weight=float(score))

        # PageRank
        try:
            scores_dict = nx.pagerank(graph, alpha=0.85, weight="weight")
            return [scores_dict[i] for i in range(n)]
        except Exception:
            return [1.0 / n] * n

    def _pick_sentences(
        self,
        cleaned_sentences: List[str],
        scores: List[float],
        k: int,
    ) -> List[int]:
        """
        Select the top-k indices by score, capped so the result stays
        under MAX_SUMMARY_WORDS total. Returns indices in ORIGINAL ORDER.
        """
        ranked = np.argsort(scores)[::-1]
        selected: List[int] = []
        word_budget = MAX_SUMMARY_WORDS

        for idx in ranked:
            if len(selected) >= k:
                break
            words_in_sent = len(cleaned_sentences[idx].split())
            if word_budget - words_in_sent < 0 and selected:
                # Already have some sentences; don't blow the budget
                continue
            selected.append(idx)
            word_budget -= words_in_sent

        # Return in original document order
        return sorted(selected)

    def _summarize_core(self, text: str, ratio: float = 0.3) -> Dict[str, Any]:
        """Run the extractive pipeline and return summary text plus source sentences."""
        if not text or not text.strip():
            return {"summary": "", "source_sentences": []}

        raw_sentences = self.preprocessor.tokenize_sentences(text)
        if len(raw_sentences) < 3:
            simple_sentences = [clean_sentence(s) for s in raw_sentences if not is_noise_sentence(s)]
            return {
                "summary": "\n\n".join(simple_sentences),
                "source_sentences": [
                    {"rank": index + 1, "source_index": index, "sentence": sentence}
                    for index, sentence in enumerate(simple_sentences)
                ],
            }

        ratio = max(0.1, min(0.5, ratio))

        cleaned, index_map = self._clean_and_filter(raw_sentences)
        if len(cleaned) < 3:
            return {
                "summary": "\n\n".join(cleaned),
                "source_sentences": [
                    {"rank": index + 1, "source_index": index_map[index], "sentence": sentence}
                    for index, sentence in enumerate(cleaned)
                ],
            }

        k = max(3, int(round(len(cleaned) * ratio)))
        avg_words = sum(len(sentence.split()) for sentence in cleaned) / len(cleaned)
        k = min(k, max(3, int(MAX_SUMMARY_WORDS / max(avg_words, 1))))

        scores = self._get_sentence_scores(cleaned)
        selected_indices = self._pick_sentences(cleaned, scores, k)

        selected = [cleaned[index] for index in selected_indices]
        source_sentences = [
            {
                "rank": position + 1,
                "source_index": index_map[index],
                "sentence": cleaned[index],
            }
            for position, index in enumerate(selected_indices)
        ]

        return {
            "summary": self._format_output(selected),
            "source_sentences": source_sentences,
        }

    def _format_output(self, sentences: List[str]) -> str:
        """
        Group sentences into short paragraphs for readability.
        Every SENTENCES_PER_PARAGRAPH sentences become one paragraph.
        """
        paragraphs: List[str] = []
        for i in range(0, len(sentences), SENTENCES_PER_PARAGRAPH):
            group = sentences[i: i + SENTENCES_PER_PARAGRAPH]
            paragraphs.append(" ".join(group))
        return "\n\n".join(paragraphs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, text: str, ratio: float = 0.3) -> str:
        """
        Extract the most important sentences from text and return them as
        clean, readable paragraphs.

        Args:
            text (str): Raw input text.
            ratio (float): Fraction of sentences to keep (0.1–0.5).
                           The hard MAX_SUMMARY_WORDS cap further limits length.

        Returns:
            str: Multi-paragraph extractive summary, free of citation noise.
        """
        return self._summarize_core(text, ratio=ratio)["summary"]

    def summarize_with_sources(self, text: str, ratio: float = 0.3) -> Dict[str, Any]:
        """Return the summary and the source sentences used to build it."""
        return self._summarize_core(text, ratio=ratio)

    def summarize_bullets(self, text: str, n: int = 5) -> List[str]:
        """
        Return exactly n key sentences as a clean list (for bullet display).

        Args:
            text (str): Raw input text.
            n (int): Number of bullet points to return.

        Returns:
            List[str]: List of clean bullet sentences.
        """
        if not text or not text.strip():
            return []

        raw_sentences = self.preprocessor.tokenize_sentences(text)
        if len(raw_sentences) < 3:
            return [clean_sentence(s) for s in raw_sentences if not is_noise_sentence(s)]

        cleaned, _ = self._clean_and_filter(raw_sentences)
        if not cleaned:
            return []

        k = min(n, len(cleaned))
        scores = self._get_sentence_scores(cleaned)
        ranked = np.argsort(scores)[::-1][:k]
        selected = sorted(ranked)
        return [cleaned[i] for i in selected]

    def get_scores(self, text: str) -> Dict[str, float]:
        """
        Return {sentence: normalized_score} for inspection.

        Args:
            text (str): Raw input text.

        Returns:
            Dict[str, float]: Mapping of sentence to 0–1 score.
        """
        if not text or not text.strip():
            return {}

        raw_sentences = self.preprocessor.tokenize_sentences(text)
        cleaned, _ = self._clean_and_filter(raw_sentences)
        if not cleaned:
            return {}

        scores = self._get_sentence_scores(cleaned)
        mn, mx = min(scores), max(scores)
        diff = mx - mn
        if diff > 1e-9:
            norm = [(s - mn) / diff for s in scores]
        else:
            norm = [1.0] * len(scores)

        return {cleaned[i]: norm[i] for i in range(len(cleaned))}


# ---------------------------------------------------------------------------
# Backward compatibility wrapper
# ---------------------------------------------------------------------------

def summarize_extractive(text: str, ratio: float = 0.3) -> str:
    """Module-level wrapper for backward compatibility."""
    return ExtractiveSummarizer().summarize(text, ratio=ratio)


from typing import Optional  # noqa: E402 (already imported above, keep for tooling)

if __name__ == "__main__":
    print("=== Running ExtractiveSummarizer Demo ===")

    sample_text = (
        "OpenAI, the artificial intelligence research laboratory, has opened a new office in London, "
        "marking its first international expansion.[1] The San Francisco-based company, which is backed "
        "by Microsoft, said the move is a vote of confidence in the UK's growing tech ecosystem.[2] "
        "Sam Altman, the chief executive of OpenAI, said the expansion is an opportunity to attract "
        "world-class talent and build safe AI systems.[citation needed] The UK government has been "
        "actively positioning the country as a global hub for AI safety research, announcing new "
        "funding.[3] OpenAI's London office will focus on research and engineering collaboration, "
        "working closely with local universities and research institutes. This expansion comes amidst "
        "intense global competition between major technology firms to dominate the generative AI sector."
    )

    summarizer = ExtractiveSummarizer()

    print("\n--- Scores ---")
    for sent, score in summarizer.get_scores(sample_text).items():
        print(f"[{score:.3f}] {sent}")

    print("\n--- Summary (ratio=0.5) ---")
    print(summarizer.summarize(sample_text, ratio=0.5))

    print("\n--- Bullets (n=3) ---")
    for b in summarizer.summarize_bullets(sample_text, n=3):
        print(f"  • {b}")

    print("\n=== Demo Complete ===")
