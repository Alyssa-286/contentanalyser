"""Lightweight topic extraction utilities based on LDA."""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.text_cleaner import clean_text


class TopicModeler:
    """Extract broad document topics using Latent Dirichlet Allocation."""

    def extract_topics(self, text: str, n_topics: int = 3, top_words: int = 5) -> List[Dict[str, Any]]:
        """Return the strongest topic labels inferred from the input text.

        Args:
            text: Raw or cleaned document text.
            n_topics: Maximum number of topics to return.
            top_words: Number of representative words per topic.

        Returns:
            A list of topic dictionaries containing a topic label, keywords, and weight.
        """
        if not text or not text.strip():
            return []

        cleaned_text = clean_text(text)
        if len(cleaned_text.split()) < 40:
            return []

        vectorizer = CountVectorizer(stop_words="english", max_features=1000)
        term_matrix = vectorizer.fit_transform([cleaned_text])
        if term_matrix.shape[1] == 0:
            return []

        topic_count = max(1, min(n_topics, 5, term_matrix.shape[1]))
        lda = LatentDirichletAllocation(
            n_components=topic_count,
            random_state=42,
            learning_method="batch",
        )
        lda.fit(term_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics: List[Dict[str, Any]] = []
        for topic_index, topic_weights in enumerate(lda.components_):
            top_indices = topic_weights.argsort()[::-1][:top_words]
            keywords = [feature_names[index] for index in top_indices]
            topics.append(
                {
                    "topic": f"Topic {topic_index + 1}",
                    "keywords": keywords,
                    "label": ", ".join(keywords[:3]),
                    "weight": round(float(topic_weights[top_indices[0]]), 4) if len(top_indices) > 0 else 0.0,
                }
            )

        return topics
