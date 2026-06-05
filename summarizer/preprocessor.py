"""
Production-quality NLP preprocessor module for text summarization.
Uses spaCy and NLTK for text cleaning, tokenization, lemmatization, stopword removal, and NER.
"""

import re
import time
from typing import List, Dict, Optional, Any, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    nltk.download("stopwords", quiet=True)
except Exception:
    pass


class TextPreprocessor:
    """
    TextPreprocessor handles all NLP preprocessing tasks including cleaning,
    tokenization, stopword removal, lemmatization, entity extraction, and pipeline orchestration.
    """

    # Class-level cache for lazy loading the spaCy model
    _nlp_cache: Optional[Any] = None
    _spacy_available: bool = True

    # Custom domain-specific stopwords to be removed in addition to standard NLTK stopwords
    DEFAULT_CUSTOM_STOPWORDS: Set[str] = {
        "article", "report", "summary", "document", "paragraph", "section",
        "page", "chapter", "author", "editor", "publisher", "date", "time"
    }

    def __init__(self, custom_stopwords: Optional[List[str]] = None) -> None:
        """
        Initialize the preprocessor with optional custom stopwords.

        Args:
            custom_stopwords (Optional[List[str]]): Additional list of domain-specific stopwords.
        """
        # Load NLTK stopwords
        try:
            self.stopwords: Set[str] = set(stopwords.words("english"))
        except Exception:
            self.stopwords = set()

        # Add default custom stopwords
        self.stopwords.update(self.DEFAULT_CUSTOM_STOPWORDS)

        # Add user-provided custom stopwords
        if custom_stopwords:
            self.stopwords.update(word.lower() for word in custom_stopwords)

    @classmethod
    def _get_nlp(cls) -> Any:
        """
        Retrieve or load the spaCy model from the class cache.

        Returns:
            Language: The loaded spaCy Language model.
        """
        if cls._nlp_cache is None and cls._spacy_available:
            try:
                import spacy

                cls._nlp_cache = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    from spacy.cli import download

                    download("en_core_web_sm")
                    cls._nlp_cache = spacy.load("en_core_web_sm")
                except Exception:
                    cls._spacy_available = False
                    cls._nlp_cache = None
            except Exception:
                cls._spacy_available = False
                cls._nlp_cache = None
        return cls._nlp_cache

    def clean(self, text: Optional[str]) -> str:
        """
        Clean the input text by removing HTML tags, URLs, special characters,
        and redundant whitespace.

        Args:
            text (Optional[str]): The raw input text.

        Returns:
            str: Cleaned and normalized text.
        """
        if not text:
            return ""

        # Remove HTML tags
        cleaned = re.sub(r"<[^>]+>", "", text)

        # Remove URLs (http, https, www)
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)

        # Remove special characters (retain letters, digits, spaces, and standard punctuation)
        cleaned = re.sub(r"[^a-zA-Z0-9\s.,!?;:\'\"()\-]", "", cleaned)

        # Normalize redundant whitespaces
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned.strip()

    def tokenize_sentences(self, text: Optional[str]) -> List[str]:
        """
        Split the text into sentences using the spaCy sentence segmenter.
        Optimized by disabling other pipeline components.

        Args:
            text (Optional[str]): The input text.

        Returns:
            List[str]: List of sentences.
        """
        if not text or not text.strip():
            return []

        nlp = self._get_nlp()
        if not nlp:
            return [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text) if sentence.strip()]

        # Disable unused components for maximum speed; parser requires tok2vec to function
        enable_pipes = []
        if "tok2vec" in nlp.pipe_names:
            enable_pipes.append("tok2vec")
        if "senter" in nlp.pipe_names:
            enable_pipes.append("senter")
        elif "parser" in nlp.pipe_names:
            enable_pipes.append("parser")
            
        with nlp.select_pipes(enable=enable_pipes):
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def tokenize_words(self, text: Optional[str]) -> List[str]:
        """
        Tokenize the text into lowercased words, stripping punctuation.

        Args:
            text (Optional[str]): The input text.

        Returns:
            List[str]: List of clean word tokens.
        """
        if not text or not text.strip():
            return []

        nlp = self._get_nlp()
        if not nlp:
            return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)]

        doc = nlp.make_doc(text)
        return [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]

    def remove_stopwords(self, tokens: Optional[List[str]]) -> List[str]:
        """
        Remove NLTK and custom stopwords from a list of tokens.

        Args:
            tokens (Optional[List[str]]): List of word tokens.

        Returns:
            List[str]: List of tokens without stopwords.
        """
        if not tokens:
            return []
        return [t for t in tokens if t.lower() not in self.stopwords]

    def lemmatize(self, tokens: Optional[List[str]]) -> List[str]:
        """
        Lemmatize a list of token strings using the spaCy lemmatizer.
        Optimized by directly constructing a spaCy Doc from tokens.

        Args:
            tokens (Optional[List[str]]): List of word tokens.

        Returns:
            List[str]: List of lemmatized tokens.
        """
        if not tokens:
            return []

        nlp = self._get_nlp()
        if not nlp:
            stemmer = PorterStemmer()
            return [stemmer.stem(token.lower()) for token in tokens]

        from spacy.tokens import Doc

        doc = Doc(nlp.vocab, words=tokens)

        needed_pipes = {"tok2vec", "tagger", "attribute_ruler", "lemmatizer"}
        for name, proc in nlp.pipeline:
            if name in needed_pipes:
                doc = proc(doc)

        result: List[str] = []
        for t in doc:
            lemma = t.lemma_.strip()
            if lemma:
                result.append(lemma.lower())
            else:
                result.append(t.text)
        return result

    def extract_entities(self, text: Optional[str]) -> Dict[str, List[str]]:
        """
        Extract named entities from the text using spaCy NER.
        Filters for PERSON, ORG, DATE, and GPE.

        Args:
            text (Optional[str]): The input text.

        Returns:
            Dict[str, List[str]]: Extracted entities classified by type.
        """
        entities: Dict[str, List[str]] = {
            "PERSON": [],
            "ORG": [],
            "DATE": [],
            "GPE": []
        }
        if not text or not text.strip():
            return entities

        nlp = self._get_nlp()
        if not nlp:
            return entities

        # Enable only NER and its dependency components
        with nlp.select_pipes(enable=["ner", "tok2vec"]):
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text.strip())
        return entities

    def full_pipeline(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Run the full preprocessing pipeline on the input text.
        Optimized by running a single spaCy doc parsing pass.

        Args:
            text (Optional[str]): The raw input text.

        Returns:
            Dict[str, Any]: Structured dictionary with intermediate results:
                - cleaned_text (str)
                - sentences (List[str])
                - tokens (List[str])
                - tokens_no_stopwords (List[str])
                - lemmas (List[str])
                - entities (Dict[str, List[str]])
        """
        # Handle empty/None input early
        if not text or not text.strip():
            return {
                "cleaned_text": "",
                "sentences": [],
                "tokens": [],
                "tokens_no_stopwords": [],
                "lemmas": [],
                "entities": {"PERSON": [], "ORG": [], "DATE": [], "GPE": []}
            }

        # 1. Clean the text
        cleaned: str = self.clean(text)

        # 2. Run single spaCy pass over the cleaned text
        nlp = self._get_nlp()
        if not nlp:
            sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', cleaned) if sentence.strip()]
            tokens = self.tokenize_words(cleaned)
            tokens_no_stopwords = self.remove_stopwords(tokens)
            lemmas = self.lemmatize(tokens)
            return {
                "cleaned_text": cleaned,
                "sentences": sentences,
                "tokens": tokens,
                "tokens_no_stopwords": tokens_no_stopwords,
                "lemmas": lemmas,
                "entities": {"PERSON": [], "ORG": [], "DATE": [], "GPE": []},
            }

        doc = nlp(cleaned)

        # 3. Extract sentences
        sentences: List[str] = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # 4. Extract words, excluding punctuation and spaces
        tokens: List[str] = [
            t.text.lower() for t in doc if not t.is_punct and not t.is_space
        ]

        # 5. Filter stopwords
        tokens_no_stopwords: List[str] = self.remove_stopwords(tokens)

        # 6. Extract lemmas for all non-punctuation tokens
        lemmas: List[str] = [
            t.lemma_.lower() if t.lemma_.strip() else t.text.lower()
            for t in doc if not t.is_punct and not t.is_space
        ]

        # 7. Extract filtered entities
        entities: Dict[str, List[str]] = {
            "PERSON": [],
            "ORG": [],
            "DATE": [],
            "GPE": []
        }
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text.strip())

        return {
            "cleaned_text": cleaned,
            "sentences": sentences,
            "tokens": tokens,
            "tokens_no_stopwords": tokens_no_stopwords,
            "lemmas": lemmas,
            "entities": entities
        }


# ==========================================
# Backward compatibility functions
# ==========================================

def tokenize_sentences(text: Optional[str]) -> List[str]:
    """
    Module-level function to tokenize text into sentences.
    Provides backward compatibility for modules importing this function directly.
    """
    return TextPreprocessor().tokenize_sentences(text)


def lemmatize_text(text: Optional[str]) -> List[str]:
    """
    Module-level function to clean, tokenize, and lemmatize text.
    Provides backward compatibility for modules importing this function directly.
    """
    if not text:
        return []
    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize_words(text)
    return preprocessor.lemmatize(tokens)


if __name__ == "__main__":
    print("=== Running Unit Tests for TextPreprocessor ===")
    preprocessor = TextPreprocessor(custom_stopwords=["customstop"])

    # 1. Test None and Empty Inputs
    print("Testing empty/None inputs...")
    assert preprocessor.clean(None) == ""
    assert preprocessor.clean("") == ""
    assert preprocessor.tokenize_sentences(None) == []
    assert preprocessor.tokenize_words(None) == []
    assert preprocessor.remove_stopwords(None) == []
    assert preprocessor.lemmatize(None) == []
    assert preprocessor.extract_entities(None) == {"PERSON": [], "ORG": [], "DATE": [], "GPE": []}
    assert preprocessor.full_pipeline(None)["cleaned_text"] == ""
    print("Success: Empty/None inputs handled gracefully.")

    # 2. Test Clean method
    print("Testing clean() method...")
    dirty_html = "<p>Welcome to <b>Google</b>. Visit http://google.com for more.</p>"
    clean_html = preprocessor.clean(dirty_html)
    assert "Welcome to Google" in clean_html
    assert "http" not in clean_html
    assert "<b>" not in clean_html
    print(f"Cleaned HTML/URL: {clean_html}")

    special_chars = "Hello!@# World$%^ &*()_+{}|:<>?"
    clean_special = preprocessor.clean(special_chars)
    assert clean_special == "Hello! World ():?"
    print(f"Cleaned Special Characters: {clean_special}")
    print("Success: clean() method verified.")

    # 3. Test tokenize_sentences
    print("Testing tokenize_sentences()...")
    abbr_text = "Dr. John Doe visited the U.S. at 10.30 AM. He met Mr. Smith. Pi is 3.14."
    sentences = preprocessor.tokenize_sentences(abbr_text)
    print(f"Parsed Sentences ({len(sentences)}):")
    for s in sentences:
        print(f"  - {s}")
    # Verify we don't break at decimal points or common abbreviations
    assert len(sentences) <= 3
    print("Success: tokenize_sentences() verified.")

    # 4. Test tokenize_words
    print("Testing tokenize_words()...")
    word_text = "The quick brown fox, jumps over the lazy dog!"
    words = preprocessor.tokenize_words(word_text)
    assert "the" in words
    assert "fox" in words
    assert "," not in words and "!" not in words
    print(f"Word tokens: {words}")
    print("Success: tokenize_words() verified.")

    # 5. Test remove_stopwords
    print("Testing remove_stopwords()...")
    mixed_tokens = ["the", "quick", "customstop", "summary", "fox"]
    filtered_tokens = preprocessor.remove_stopwords(mixed_tokens)
    assert "the" not in filtered_tokens
    assert "customstop" not in filtered_tokens
    assert "summary" not in filtered_tokens
    assert "quick" in filtered_tokens
    print(f"Filtered tokens: {filtered_tokens}")
    print("Success: remove_stopwords() verified.")

    # 6. Test lemmatize
    print("Testing lemmatize()...")
    verbs = ["running", "ran", "runs", "easily", "dogs"]
    lemmas = preprocessor.lemmatize(verbs)
    assert "run" in lemmas
    assert "dog" in lemmas
    print(f"Lemmas: {lemmas}")
    print("Success: lemmatize() verified.")

    # 7. Test extract_entities
    print("Testing extract_entities()...")
    ner_text = "Sundar Pichai is the CEO of Google. He visited Paris on January 1, 2026."
    entities = preprocessor.extract_entities(ner_text)
    print(f"Entities: {entities}")
    assert "Sundar Pichai" in entities["PERSON"]
    assert "Google" in entities["ORG"]
    assert "Paris" in entities["GPE"]
    assert "January 1, 2026" in entities["DATE"]
    print("Success: extract_entities() verified.")

    # 8. Test full_pipeline
    print("Testing full_pipeline()...")
    pipeline_res = preprocessor.full_pipeline(ner_text)
    assert len(pipeline_res["sentences"]) > 0
    assert len(pipeline_res["tokens"]) > 0
    assert len(pipeline_res["lemmas"]) > 0
    assert len(pipeline_res["entities"]["PERSON"]) > 0
    print("Success: full_pipeline() output structure and correctness verified.")

    # 9. Performance Benchmark (10,000 words under 3 seconds)
    print("\nRunning Performance Benchmark (10,000 words)...")
    base_text = (
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals. "
        "Sundar Pichai is the CEO of Google. He visited Paris on January 1, 2026. "
        "The quick brown fox jumps over the lazy dog. "
    )
    # Generate ~10,000 words of text
    words_in_base = len(base_text.split())
    multiplier = 10000 // words_in_base + 1
    large_text = base_text * multiplier
    actual_word_count = len(large_text.split())
    print(f"Generated benchmark text of {actual_word_count} words.")

    start_time = time.time()
    result = preprocessor.full_pipeline(large_text)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Processed {actual_word_count} words in {elapsed_time:.4f} seconds.")
    print(f"Sentences extracted: {len(result['sentences'])}")
    print(f"Tokens extracted: {len(result['tokens'])}")

    # Assert that runtime is less than 3 seconds
    assert elapsed_time < 3.0, f"Benchmark failed: took {elapsed_time:.4f} seconds (limit is 3.0 seconds)"
    print("Success: Performance Benchmark passed successfully!")
    print("=== All tests passed! ===")
