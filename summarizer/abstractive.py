"""
Abstractive summarizer using Hugging Face's BART model.
Includes lazy loading, automatic device detection, smart map-reduce chunking for long documents,
batch inference, and progress tracking.
"""

import time
from typing import List, Optional
from tqdm import tqdm

from config import MODEL_NAME


class AbstractiveSummarizer:
    """
    AbstractiveSummarizer wraps Hugging Face's facebook/bart-large-cnn model
    to generate fluent summaries of text, supporting chunking and GPU acceleration.
    """

    # Class-level caches
    _model: Optional[object] = None
    _tokenizer: Optional[object] = None
    _device_str: Optional[str] = None
    _available: bool = True

    def __init__(self, device: str = "auto") -> None:
        """
        Initialize the AbstractiveSummarizer, loading the model and tokenizer
        if they are not already cached.

        Args:
            device (str): Device to run inference on ('auto', 'cuda', 'mps', 'cpu').
                          Defaults to 'auto'.
        """
        if AbstractiveSummarizer._model is None or AbstractiveSummarizer._tokenizer is None:
            try:
                import torch
                from transformers import BartForConditionalGeneration, BartTokenizer

                # Auto-detect device
                if device == "auto":
                    if torch.cuda.is_available():
                        detected_device = "cuda"
                    elif torch.backends.mps.is_available():
                        detected_device = "mps"
                    else:
                        detected_device = "cpu"
                else:
                    detected_device = device

                print(f"Loading AbstractiveSummarizer model onto device: {detected_device}...")

                tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
                model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
                device_obj = torch.device(detected_device)
                model = model.to(device_obj)

                AbstractiveSummarizer._tokenizer = tokenizer
                AbstractiveSummarizer._model = model
                AbstractiveSummarizer._device_str = detected_device
            except Exception as exc:
                print(f"Abstractive model unavailable, using fallback summarizer: {exc}")
                AbstractiveSummarizer._available = False
                AbstractiveSummarizer._tokenizer = None
                AbstractiveSummarizer._model = None
                AbstractiveSummarizer._device_str = "fallback"
        else:
            print(f"Using cached AbstractiveSummarizer on device: {AbstractiveSummarizer._device_str}")

        self.tokenizer = AbstractiveSummarizer._tokenizer
        self.model = AbstractiveSummarizer._model
        self.device_str = AbstractiveSummarizer._device_str

    def _summarize_single_chunk(self, text: str, max_length: int, min_length: int) -> str:
        """
        Summarize a single text chunk that is guaranteed to be under the token limit.
        """
        if not self._available or not self.tokenizer or not self.model:
            words = text.split()
            if len(words) <= 40:
                return text.strip()
            return " ".join(words[:max(40, min(len(words), max_length))]).strip()

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            max_length=max_length,
            min_length=min_length
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        preferred_output: str = "High Quality",
    ) -> str:
        """
        Generate an abstractive summary of the text. Supports long texts by
        chunking with overlap and utilizing map-reduce. Falls back to extractive
        summarization if BART fails.

        Args:
            text (str): Input text to summarize.
            max_length (int): Maximum length of the generated summary.
            min_length (int): Minimum length of the generated summary.

        Returns:
            str: Generated summary.
        """
        if not text or not text.strip():
            return ""

        if not self._available or not self.tokenizer or not self.model:
            from summarizer.preprocessor import TextPreprocessor
            from utils.text_cleaner import clean_sentence, is_noise_sentence

            preprocessor = TextPreprocessor()
            sentences = preprocessor.tokenize_sentences(text)
            cleaned = [clean_sentence(sentence) for sentence in sentences if not is_noise_sentence(sentence)]
            if not cleaned:
                return text[:500]

            if preferred_output == "Fast Summary":
                chosen = cleaned[:1]
            else:
                chosen = cleaned[:4]

            fused = []
            seen = set()
            for sentence in chosen:
                normalized = sentence.rstrip(". ")
                if normalized and normalized not in seen:
                    fused.append(normalized)
                    seen.add(normalized)

            if preferred_output == "Fast Summary":
                return ". ".join(fused[:2]).strip() + ("." if fused else "")

            return ". ".join(fused).strip() + ("." if fused else "")

        try:
            # Tokenize and get count
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            n_tokens = len(encoded)

            # BART max position embedding is 1024 tokens
            if n_tokens > 1024:
                # Map step: chunk into overlapping segments
                max_chunk_size = 950  # Leave safety margin for special tokens
                overlap = 50
                chunk_summaries: List[str] = []

                i = 0
                while i < n_tokens:
                    chunk_token_ids = encoded[i : i + max_chunk_size]
                    chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                    
                    # Summarize the chunk
                    # Adjust dynamic chunk limit to prevent warnings
                    chunk_word_count = len(chunk_text.split())
                    dynamic_max = min(max_length, max(min_length + 10, chunk_word_count // 2))
                    
                    summary = self._summarize_single_chunk(chunk_text, max_length=dynamic_max, min_length=min_length)
                    if summary:
                        chunk_summaries.append(summary)

                    if i + max_chunk_size >= n_tokens:
                        break
                    i += (max_chunk_size - overlap)

                # Reduce step: concatenate and summarize again
                concatenated_text = " ".join(chunk_summaries)
                return self.summarize(
                    concatenated_text,
                    max_length=max_length,
                    min_length=min_length,
                    preferred_output=preferred_output,
                )
            else:
                return self._summarize_single_chunk(text, max_length=max_length, min_length=min_length)

        except Exception as e:
            print(f"BART summarization failed: {e}. Falling back to ExtractiveSummarizer.")
            try:
                from summarizer.extractive import ExtractiveSummarizer
                ext_summarizer = ExtractiveSummarizer()
                # Pick a fallback extractive ratio based on preferred_output so
                # 'Fast Summary' yields a shorter result and 'High Quality' yields a fuller one.
                fallback_ratio = 0.12 if preferred_output == "Fast Summary" else 0.35
                return ext_summarizer.summarize(text, ratio=fallback_ratio)
            except Exception as ext_err:
                print(f"Extractive fallback failed: {ext_err}")
                return text[:500]  # Ultimate fallback to slicing text

    def summarize_batch(self, texts: List[str]) -> List[str]:
        """
        Process multiple texts efficiently using batch inference.
        Shows a tqdm progress bar during processing.

        Args:
            texts (List[str]): List of documents to summarize.

        Returns:
            List[str]: List of summaries.
        """
        if not texts:
            return []

        results: List[str] = []
        batch_size = 4

        # Configure padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for i in tqdm(range(0, len(texts), batch_size), desc="Summarizing batches"):
            batch_texts = texts[i : i + batch_size]
            
            # To handle extremely long documents in batch, we truncate to 1024 tokens
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    max_length=150,
                    min_length=40
                )
                
                decoded = [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in summary_ids]
                results.extend(decoded)
            except Exception as e:
                print(f"Batch generation failed for index {i}: {e}. Processing items individually.")
                for t in batch_texts:
                    results.append(self.summarize(t))

        return results

    def estimate_time(self, text: str) -> float:
        """
        Estimate generation time in seconds based on text length and hardware device.

        Args:
            text (str): Input text.

        Returns:
            float: Estimated execution time in seconds.
        """
        if not text:
            return 0.0
        
        words = len(text.split())
        approx_tokens = int(words * 1.3)

        # CUDA/MPS are significantly faster than CPU
        if "cuda" in self.device_str or "mps" in self.device_str:
            return max(0.5, approx_tokens * 0.006)
        else:
            return max(1.5, approx_tokens * 0.05)


if __name__ == "__main__":
    print("=== Running AbstractiveSummarizer Demo ===")
    
    # Simple test article
    sample_text = (
        "OpenAI has released its new reasoning model, GPT-o1, which represents a step forward in "
        "solving complex scientific, mathematical, and coding problems. Unlike previous models "
        "that respond immediately, GPT-o1 is designed to spend more time thinking before answering. "
        "During this thinking phase, the model refines its thought process, tries different strategies, "
        "and recognizes its errors. In tests, the model performed similarly to PhD students on "
        "challenging benchmark tasks. OpenAI plans to roll out this model to all ChatGPT Plus users "
        "starting this week, indicating a major shift in how AI models approach logic and reasoning."
    )

    # Initialize summarizer (auto device detection)
    summarizer = AbstractiveSummarizer()

    # Estimate time
    est_sec = summarizer.estimate_time(sample_text)
    print(f"Estimated processing time: {est_sec:.2f} seconds")

    # Run summarize
    start = time.time()
    summary = summarizer.summarize(sample_text)
    elapsed = time.time() - start
    print(f"\n[Generated Summary] (took {elapsed:.2f}s):")
    print(summary)

    # Run batch summarize
    print("\n--- Testing Batch Summarization ---")
    batch_texts = [sample_text, sample_text[:100] + " is an extremely short document for testing."]
    batch_summaries = summarizer.summarize_batch(batch_texts)
    for idx, s in enumerate(batch_summaries, 1):
        print(f"Summary {idx}: {s}")

    print("\n=== Demo Complete ===")
