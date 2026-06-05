"""
Flask API application serving the Smart Content Summarization System.
Handles JSON requests and file uploads (PDF/TXT), returning summaries, keywords, and entities.
"""

import os
import time
import tempfile
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from flask import Flask, request, jsonify, Response

from config import MAX_INPUT_CHARS, DEFAULT_SUMMARY_RATIO, MODEL_NAME
from utils.text_cleaner import clean_text
from utils.file_parser import parse_file

# Global models initialized at startup
preprocessor = None
extractive_summarizer = None
abstractive_summarizer = None
keyword_extractor = None
models_loaded = False

app = Flask(__name__)

# Configure maximum file upload size limit (5MB as requested)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def load_models() -> None:
    """
    Load all models and preprocessors in memory at startup.
    """
    global preprocessor, extractive_summarizer, abstractive_summarizer, keyword_extractor, models_loaded
    if not models_loaded:
        print("--- Loading all models at startup ---")
        from summarizer.preprocessor import TextPreprocessor
        from summarizer.extractive import ExtractiveSummarizer
        from summarizer.abstractive import AbstractiveSummarizer
        from summarizer.keywords import KeywordExtractor

        preprocessor = TextPreprocessor()
        extractive_summarizer = ExtractiveSummarizer(preprocessor=preprocessor)
        abstractive_summarizer = AbstractiveSummarizer()
        keyword_extractor = KeywordExtractor()
        models_loaded = True
        print("--- All models loaded successfully ---")


# Register startup loader based on Flask version capability
if hasattr(app, "before_first_request"):
    @app.before_first_request
    def startup_load() -> None:
        load_models()
else:
    # Flask 2.3+ fallback: load immediately upon module import
    load_models()


@app.before_request
def start_timer() -> None:
    """
    Start request timer before routing.
    """
    request.start_time = time.time()


@app.before_request
def handle_options() -> Optional[Response]:
    """
    Intercept and handle CORS preflight OPTIONS requests.
    """
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return response
    return None


@app.after_request
def add_cors_headers(response: Response) -> Response:
    """
    Manually add CORS headers to response to support frontend applications.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.after_request
def log_request(response: Response) -> Response:
    """
    Request logging with timestamp, endpoint, status code, and response time.
    """
    if hasattr(request, "start_time"):
        elapsed = time.time() - request.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {request.method} {request.path} - Status: {response.status_code} - Time: {elapsed:.4f}s")
        if response.status_code == 400:
            print(f"  [400 Error Body]: {response.get_data(as_text=True)}")
    return response


@app.errorhandler(413)
def file_too_large(e: Any) -> Tuple[Response, int]:
    """
    Handle payload too large exceptions.
    """
    return jsonify({"error": "File size exceeds maximum limit of 5MB."}), 413


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """
    Health check endpoint returning model loading status.

    Returns:
        Tuple[Response, int]: JSON status and HTTP code.
    """
    return jsonify({
        "status": "ok",
        "models_loaded": models_loaded
    }), 200


def _process_summarization_pipeline(
    text: str,
    mode: str,
    ratio: float
) -> Tuple[Response, int]:
    """
    Helper function to run the cleaning, summarization, entity extraction, and keyword pipeline.
    """
    global preprocessor, extractive_summarizer, abstractive_summarizer, keyword_extractor

    # 0. Check for URL input and resolve
    text_strip = text.strip()
    if text_strip.lower().startswith("http://") or text_strip.lower().startswith("https://"):
        try:
            if "youtube.com" in text_strip.lower() or "youtu.be" in text_strip.lower():
                from utils.youtube_parser import YouTubeParser
                yt_parser = YouTubeParser()
                text = yt_parser.get_transcript(text_strip)
            else:
                from utils.file_parser import FileParser
                f_parser = FileParser()
                text = f_parser.parse_url(text_strip)
        except ValueError as e:
            # ValueError messages from our parsers are already user-friendly — pass them through directly
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to resolve URL: {str(e)}"}), 400

    # Validate input length
    original_char_count = len(text)
    if original_char_count > MAX_INPUT_CHARS:
        return jsonify({
            "error": f"Input text length ({original_char_count} chars) exceeds maximum limit of {MAX_INPUT_CHARS} characters."
        }), 400

    # 1. Clean Text
    cleaned_text = clean_text(text)
    if not cleaned_text or not cleaned_text.strip():
        return jsonify({"error": "Input contains no extractable text after cleaning."}), 400

    original_word_count = len(cleaned_text.split())

    # 2. Summarize
    summary = ""
    try:
        if mode == "abstractive":
            summary = abstractive_summarizer.summarize(cleaned_text)
        else:
            summary = extractive_summarizer.summarize(cleaned_text, ratio=ratio)
    except Exception as e:
        return jsonify({"error": f"Summarization process failed: {str(e)}"}), 500

    # 3. Extract Keywords (Top 10)
    keywords = []
    if keyword_extractor:
        try:
            # Return list of keyword strings
            raw_kws = keyword_extractor.extract_tfidf(cleaned_text, n=10)
            keywords = [kw for kw, score in raw_kws]
        except Exception:
            pass

    # 4. Extract Entities
    entities = {"PERSON": [], "ORG": [], "DATE": [], "GPE": []}
    if preprocessor:
        try:
            entities = preprocessor.extract_entities(cleaned_text)
        except Exception:
            pass

    # 5. Calculate Compression Ratio
    summary_word_count = len(summary.split())
    if original_word_count > 0:
        reduction = max(0, int(round((1 - (summary_word_count / original_word_count)) * 100)))
    else:
        reduction = 0
    compression_str = f"{reduction}% reduction"

    return jsonify({
        "summary": summary,
        "keywords": keywords,
        "entities": entities,
        "word_count": original_word_count,
        "compression": compression_str
    }), 200


@app.route("/summarize", methods=["POST"])
def summarize() -> Tuple[Response, int]:
    """
    Summarize raw JSON text inputs.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    text = data.get("text", "")
    if not text or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing or empty required field 'text'."}), 400

    mode = data.get("mode", "extractive").lower()
    if mode not in ("extractive", "abstractive"):
        return jsonify({"error": "Invalid mode. Use 'extractive' or 'abstractive'."}), 400

    try:
        ratio = float(data.get("ratio", DEFAULT_SUMMARY_RATIO))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid ratio value. Must be a float."}), 400

    if not (0.1 <= ratio <= 0.5):
        return jsonify({"error": "Ratio must be between 0.1 and 0.5."}), 400

    return _process_summarization_pipeline(text, mode, ratio)


@app.route("/upload", methods=["POST"])
def upload() -> Tuple[Response, int]:
    """
    Ingest TXT or PDF file and return summarization results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field found in the request."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"error": "Selected file is empty."}), 400

    # Retrieve request parameters from form
    mode = request.form.get("mode", "extractive").lower()
    if mode not in ("extractive", "abstractive"):
        return jsonify({"error": "Invalid mode. Use 'extractive' or 'abstractive'."}), 400

    try:
        ratio = float(request.form.get("ratio", str(DEFAULT_SUMMARY_RATIO)))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid ratio value. Must be a float."}), 400

    if not (0.1 <= ratio <= 0.5):
        return jsonify({"error": "Ratio must be between 0.1 and 0.5."}), 400

    suffix = os.path.splitext(uploaded_file.filename)[1].lower()
    if suffix not in (".txt", ".pdf"):
        return jsonify({"error": "Unsupported file format. Only .txt and .pdf are supported."}), 400

    # Save to a temporary file to parse
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        uploaded_file.save(temp_path)
        extracted_text = parse_file(temp_path)
    except Exception as e:
        return jsonify({"error": f"Failed to parse uploaded file: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not extracted_text or not extracted_text.strip():
        return jsonify({"error": "Failed to extract text from file."}), 400

    return _process_summarization_pipeline(extracted_text, mode, ratio)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
