"""ROUGE evaluation helpers for reference-based summarization scoring."""

from __future__ import annotations

from typing import Dict

from rouge_score import rouge_scorer


def compute_rouge_scores(reference_summary: str, generated_summary: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores for two summaries.

    Args:
        reference_summary: Human-written reference summary.
        generated_summary: Model-generated summary.

    Returns:
        A nested dictionary containing precision, recall, and F1 for each ROUGE metric.
    """
    if not reference_summary or not reference_summary.strip():
        return {}
    if not generated_summary or not generated_summary.strip():
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)

    return {
        metric: {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "fmeasure": round(score.fmeasure, 4),
        }
        for metric, score in scores.items()
    }
