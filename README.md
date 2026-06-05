# Content-analyser

An NLP Activity-Based Learning project for end-to-end content summarization.
The system ingests raw text, PDF/TXT files, web URLs, and YouTube transcripts,
then produces summaries using both extractive and abstractive NLP pipelines.

## Project Goal

The aim of this project is to demonstrate a practical content understanding
pipeline with clear NLP depth and a usable full-stack interface. It combines:

- Flask for the backend API
- Streamlit for the interactive dashboard
- NLTK with optional spaCy for preprocessing and named entity extraction
- TF-IDF for keyword extraction
- TextRank-style extractive summarization
- BART-based abstractive summarization

This makes the project suitable for academic evaluation because it shows both
classic NLP methods and transformer-based summarization in one system.

## Current Features

- Summarize pasted text through a Flask API
- Upload and summarize `.txt` and `.pdf` files
- Summarize content from web URLs
- Extract YouTube transcripts and summarize them
- Generate extractive summaries using a TextRank-style pipeline
- Generate abstractive summaries using `facebook/bart-large-cnn`
- Extract keywords with TF-IDF
- Detect named entities such as PERSON, ORG, DATE, and GPE
- Estimate broad topics from the document with LDA
- Evaluate generated summaries against a reference summary using ROUGE
- Display results in a Streamlit dashboard
- Download the generated summary as a text file

## Why This Project Matters

This project is a strong NLP learning artifact because it shows the full flow
of a real system:

1. Input ingestion from multiple content sources
2. Text cleaning and normalization
3. NLP preprocessing
4. Summarization with two different paradigms
5. Keyword and entity extraction
6. Frontend presentation and user interaction

That makes it more than a model demo. It is an applied NLP application.

## Repository Structure

```text
Content-analyser/
├── app.py
├── config.py
├── requirements.txt
├── streamlit_app.py
├── summarizer/
│   ├── __init__.py
│   ├── abstractive.py
│   ├── extractive.py
│   ├── keywords.py
│   └── preprocessor.py
├── utils/
│   ├── __init__.py
│   ├── file_parser.py
│   ├── text_cleaner.py
│   └── youtube_parser.py
├── summary_test.txt
├── test_import.py
└── test_segmentation.py
```

## System Architecture

The application uses a simple two-layer architecture:

- `streamlit_app.py` is the user interface.
- `app.py` is the backend service that performs parsing and NLP processing.

The backend pipeline works like this:

1. Receive text, file, URL, or YouTube input
2. Parse and normalize the content
3. Clean the text
4. Run extractive or abstractive summarization
5. Extract keywords and entities
6. Return structured JSON to the UI

## NLP Pipeline Overview

### Preprocessing

The preprocessing layer uses spaCy and NLTK to:

- Remove HTML, URLs, and noisy artifacts
- Split text into sentences
- Tokenize words
- Remove stopwords
- Lemmatize tokens
- Extract named entities

### Extractive Summarization

The extractive pipeline in `summarizer/extractive.py` uses a TextRank-style
approach:

- Sentence scoring is based on TF-IDF cosine similarity
- Important sentences are ranked with PageRank
- Noise sentences are filtered before scoring
- The final summary is formatted into readable paragraphs
- Source sentences are preserved for explainability in the UI

### Abstractive Summarization

The abstractive pipeline in `summarizer/abstractive.py` uses Hugging Face BART:

- `facebook/bart-large-cnn` is loaded through `transformers`
- The model generates fluent natural-language summaries
- Long texts are chunked and summarized in stages
- If generation fails, the system falls back to extractive summarization

### Keyword Extraction

The keyword module uses TF-IDF over sentence-level text to return the most
relevant terms in the document.

### Topic Modeling

The topic layer uses Latent Dirichlet Allocation to extract a few broad topics
from the cleaned document. This helps the UI show what the document is about
before the summary is read.

## Setup Instructions

### 1. Create a Python environment

Use your preferred environment manager, then install dependencies:

```bash
pip install -r requirements.txt
```

If you want the richer spaCy-backed preprocessing path, also install the
spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

If spaCy or `torch` is unavailable, the app still runs with a lighter fallback
pipeline.

### 3. Run the Flask backend

```bash
python app.py
```

The backend runs on `http://localhost:5000`.

### 4. Run the Streamlit frontend

In a second terminal:

```bash
streamlit run streamlit_app.py
```

### 5. Quick smoke test

The repo includes two simple validation scripts:

```bash
python test_import.py
python test_segmentation.py
```

## API Endpoints

### `GET /health`

Returns service status and whether the NLP models are loaded.

Example response:

```json
{
  "status": "ok",
  "models_loaded": true
}
```

### `POST /summarize`

Accepts JSON input:

```json
{
  "text": "Your input text",
  "mode": "extractive",
  "ratio": 0.2,
  "reference_summary": "Optional human-written reference summary"
}
```

### `POST /upload`

Accepts a file upload for `.txt` or `.pdf` documents.

## Output Format

The backend returns a JSON payload with:

- `summary`
- `keywords`
- `entities`
- `topics`
- `source_sentences` for extractive mode
- `rouge` when a reference summary is provided
- `word_count`
- `compression`

This makes the frontend easy to extend for later evaluation features.

## Evaluation Plan

The next phase of the project will add formal summarization evaluation using
reference summaries and ROUGE metrics. That will help quantify model quality
instead of only showing generated output.

Planned evaluation items:

- ROUGE-1, ROUGE-2, and ROUGE-L
- Reference-summary benchmarking dataset
- Side-by-side comparison of generated and reference summaries
- Topic-level qualitative review for sample documents

## Demo Dataset Plan

For an academic demo, the project should include a small curated benchmark set
with at least five documents covering different content types:

- A news article
- A scientific article or report
- A blog post or opinion piece
- A PDF document
- A YouTube transcript

Each document should ideally include:

- Source text
- Reference summary
- Expected keywords
- A short note on what the summary should capture

## Roadmap

### Phase 1: Presentation and Credibility

- Write the README and project documentation
- Show extractive and abstractive summaries side by side
- Add ROUGE-based evaluation with reference summaries
- Curate a small demo dataset for benchmarking

### Phase 2: NLP Depth

- Add topic extraction or topic modeling
- Improve abstractive summarization for long documents using overlap-aware chunking
- Add source-sentence highlighting for explainability

### Phase 3: Scalability and Polish

- Add caching for repeated URLs and transcripts
- Add model selection for fast vs high-quality summarization
- Improve timeout handling and graceful fallbacks
- Move heavy processing to background jobs if needed

## Dependencies

Main libraries used by the project:

- Flask
- streamlit
- transformers
- torch
- spaCy
- nltk
- scikit-learn
- PyMuPDF
- youtube-transcript-api
- beautifulsoup4
- requests

## Notes

- The abstractive summarizer is expensive to run on CPU and may take time to
  load the first time.
- The YouTube transcript path requires subtitles or captions to be available.
- URL summarization depends on the target website allowing scraping.

## Academic Framing

If you are presenting this as an NLP ABL project, the strongest narrative is:

- multi-source document ingestion
- dual summarization paradigms
- explainable NLP enrichment with keywords and entities
- formal evaluation with ROUGE
- interface design that makes the NLP pipeline visible

That combination makes the project both practical and academically defensible.
