"""
Configuration module for the Smart Content Summarization System.
Defines system-wide constants, limits, and configuration values.
"""

from typing import Final

# Maximum length of text (in characters) allowed for summarization requests
MAX_INPUT_CHARS: Final[int] = 100000

# Default ratio of sentences to extract for extractive summarization
DEFAULT_SUMMARY_RATIO: Final[float] = 0.2

# The Hugging Face transformer model to be used for abstractive summarization
MODEL_NAME: Final[str] = "facebook/bart-large-cnn"
