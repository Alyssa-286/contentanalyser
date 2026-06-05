"""
Keyword extractor using TF-IDF and sentence context matching.
Provides tools to extract keywords, locate context sentences, and highlight them in HTML.
"""

from summarizer.preprocessor import tokenize_sentences, lemmatize_text
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordExtractor:
    """
    KeywordExtractor handles keyword extraction using TF-IDF across sentences,
    finds context sentences, and highlights keywords in HTML.
    """

    def extract_tfidf(self, text: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract the top N keywords and their TF-IDF scores from the text.
        Fitted over sentences as documents.

        Args:
            text (str): Raw input text.
            n (int): Number of top keywords to return. Defaults to 10.

        Returns:
            List[Tuple[str, float]]: Top N keywords with scores, sorted descending.
        """
        if not text or not text.strip():
            return []

        # Split text into sentences
        sentences = tokenize_sentences(text)
        if not sentences:
            return []

        # Preprocess each sentence (lemmatization)
        processed_sentences: List[str] = []
        for sent in sentences:
            lemmas = lemmatize_text(sent)
            if lemmas:
                processed_sentences.append(" ".join(lemmas))

        if not processed_sentences:
            return []

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        except ValueError:
            return []

        feature_names = np.array(vectorizer.get_feature_names_out())
        # Sum TF-IDF scores for each word across all sentences to find global relevance
        scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()

        # Sort indices descending by score
        sorted_indices = np.argsort(scores)[::-1]

        # Extract top-n results
        results = [
            (str(feature_names[idx]), float(scores[idx]))
            for idx in sorted_indices[:n]
        ]
        return results

    def extract_with_context(self, text: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract the top N keywords and identify the sentence where each keyword
        has the highest TF-IDF score.

        Args:
            text (str): Raw input text.
            n (int): Number of top keywords to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: Each item has 'keyword', 'score', and 'sentence'.
        """
        if not text or not text.strip():
            return []

        sentences = tokenize_sentences(text)
        if not sentences:
            return []

        processed_sentences: List[str] = []
        for sent in sentences:
            lemmas = lemmatize_text(sent)
            # Retain empty strings to maintain index matching with sentences
            processed_sentences.append(" ".join(lemmas) if lemmas else "")

        # Get top keywords with scores first
        top_kws = self.extract_tfidf(text, n=n)
        if not top_kws:
            return []

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(processed_sentences)
            vocab = vectorizer.vocabulary_
        except ValueError:
            vocab = {}
            tfidf_matrix = None

        results = []
        for kw, score in top_kws:
            best_sentence = ""
            # Find the sentence index where the keyword scores the highest
            if vocab and kw in vocab and tfidf_matrix is not None:
                col_idx = vocab[kw]
                # Get the column scores across all sentences
                kw_scores = tfidf_matrix[:, col_idx].toarray().ravel()
                best_idx = int(np.argmax(kw_scores))
                # Ensure the sentence has some relevance; otherwise fallback
                if kw_scores[best_idx] > 0.0:
                    best_sentence = sentences[best_idx].strip()

            # Fallback: scan sentences linearly to find the first match containing the keyword
            if not best_sentence:
                for sent in sentences:
                    if re.search(rf"\b{re.escape(kw)}\b", sent, re.IGNORECASE):
                        best_sentence = sent.strip()
                        break

            # Secondary fallback if still empty
            if not best_sentence and sentences:
                best_sentence = sentences[0].strip()

            results.append({
                "keyword": kw,
                "score": round(score, 4),
                "sentence": best_sentence
            })

        return results

    def highlight_html(self, text: str, keywords: List[str]) -> str:
        """
        Wrap occurrences of the keywords in the text with <mark class="kw"> tags.
        Keywords are matched case-insensitively but original case is preserved.

        Args:
            text (str): Raw input text.
            keywords (List[str]): List of keywords to highlight.

        Returns:
            str: Highlighted text.
        """
        if not text or not keywords:
            return text or ""

        # Sort keywords by length in descending order to avoid overlapping substring replacements
        sorted_kws = sorted(keywords, key=len, reverse=True)
        highlighted = text

        for kw in sorted_kws:
            if not kw.strip():
                continue
            # Match whole words only
            pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
            highlighted = pattern.sub(r'<mark class="kw">\1</mark>', highlighted)

        return highlighted


# Backward compatibility wrapper for step 65
def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Module-level function for backward compatibility.
    """
    extractor = KeywordExtractor()
    kws_with_scores = extractor.extract_tfidf(text, n=top_n)
    return [word for word, score in kws_with_scores]
