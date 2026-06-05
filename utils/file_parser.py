"""
File parser utility to ingest and extract raw text from various sources:
Plain Text (.txt), PDF documents (.pdf), and Web URLs.
"""

import os
import re
from typing import Dict, Optional, Any
import chardet
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF


class FileParser:
    """
    FileParser is responsible for ingesting text from PDF documents,
    plain text files, and web page URLs.
    """

    def __init__(self) -> None:
        self.last_title: Optional[str] = None

    def parse_pdf(self, path: str) -> str:
        """
        Extract text page by page from a PDF document using PyMuPDF.
        Skips image-only pages and cleans extracted text.

        Args:
            path (str): Path to the PDF file.

        Returns:
            str: Extracted and cleaned text.

        Raises:
            ValueError: If the file does not exist or cannot be parsed.
        """
        if not os.path.exists(path):
            raise ValueError(f"PDF file not found at: {path}")

        extracted_pages = []
        try:
            with fitz.open(path) as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    
                    # Skip image-only pages (empty or extremely short text)
                    if not page_text or len(page_text.strip()) < 10:
                        continue

                    cleaned_page = self._clean_page_layout(page_text)
                    if cleaned_page:
                        extracted_pages.append(cleaned_page)

        except Exception as e:
            raise ValueError(f"Failed to parse PDF file '{path}': {str(e)}")

        full_text = "\n\n".join(extracted_pages).strip()
        if not full_text:
            raise ValueError(f"No extractable text found in PDF: {path}")

        return full_text

    def parse_txt(self, path: str) -> str:
        """
        Read content from a plain text file using chardet encoding detection.
        Normalizes line endings to standard newlines.

        Args:
            path (str): Path to the text file.

        Returns:
            str: Extracted and normalized text.

        Raises:
            ValueError: If the file does not exist or cannot be parsed.
        """
        if not os.path.exists(path):
            raise ValueError(f"Text file not found at: {path}")

        try:
            with open(path, "rb") as f:
                raw_data = f.read()

            if not raw_data:
                return ""

            # Auto-detect encoding
            result = chardet.detect(raw_data)
            encoding = result.get("encoding") or "utf-8"

            # Decode with error replacement to prevent crashes on invalid bytes
            text = raw_data.decode(encoding, errors="replace")

            # Normalize line endings
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return text

        except Exception as e:
            raise ValueError(f"Failed to read text file '{path}': {str(e)}")

    def parse_url(self, url: str) -> str:
        """
        Fetch web page content, filter navigation/footers/ads, and return
        clean article body text.

        Args:
            url (str): Web URL.

        Returns:
            str: Extracted article body text.

        Raises:
            ValueError: If URL fetch or parsing fails.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            raise ValueError(
                f"Failed to fetch content from URL '{url}': {str(e)}. "
                "This website may be blocking automated scrapers (e.g. Cloudflare / anti-bot protection). "
                "If this persists, please copy-paste the text content directly in the 'Paste Text' tab."
            )

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            # Save the title to metadata cache
            if soup.title and soup.title.string:
                self.last_title = soup.title.string.strip()
            else:
                self.last_title = None

            # Remove clearly non-content tags from <head> only
            head = soup.find("head")
            if head:
                for tag in head(["link", "script", "style", "meta"]):
                    tag.decompose()

            # Remove non-content tags from the body (but NOT <link> which Wikipedia uses
            # inside paragraphs as inline metadata — removing it causes BS4 to misparse the DOM)
            body_unwanted = ["nav", "footer", "script", "style", "form", "iframe", "noscript", "svg", "button"]
            for tag in soup(body_unwanted):
                tag.decompose()

            # Remove navigation/ad/comment elements by class or id —
            # CRITICAL: never remove structural container tags (html, body, main, article, section)
            # because their class attributes may accidentally match our bad_patterns (e.g. Wikipedia's
            # <html> has class "vector-feature-main-menu-pinned-disabled" which contains "menu").
            SAFE_TAGS = {"html", "body", "main", "article", "section"}
            bad_patterns = re.compile(
                r"\bnav\b|footer|sidebar|widget|banner|adsense|social-share|comment|popup|cookie|promo",
                re.IGNORECASE
            )
            for tag in list(soup.find_all(attrs={"class": bad_patterns})):
                if tag.name not in SAFE_TAGS:
                    tag.decompose()
            for tag in list(soup.find_all(attrs={"id": bad_patterns})):
                if tag.name not in SAFE_TAGS:
                    tag.decompose()

            # Locate central text container with priority order:
            # 1. <article> tag  2. Known content ids (Wikipedia, news sites)
            # 3. <main> tag     4. <body> fallback
            container = soup.find("article")
            if not container:
                container = (
                    soup.find(id="mw-content-text") or      # Wikipedia
                    soup.find(id="content-body") or
                    soup.find(id="article-body") or
                    soup.find(id="story-body") or
                    soup.find(class_=re.compile(
                        r"article.?body|story.?body|post.?content|entry.?content", re.I
                    ))
                )
            if not container:
                container = soup.find("main")
            if not container:
                container = soup.find("body")
            if not container:
                container = soup

            # Extract from paragraphs (min 40 chars to skip nav/caption noise)
            paragraphs = container.find_all("p")
            if paragraphs:
                text_list = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40]
                text = "\n\n".join(text_list)
            else:
                text = container.get_text(separator="\n").strip()

            # Ultimate fallback to full soup text
            if not text:
                text = soup.get_text(separator="\n").strip()

            # Clean multiple newlines and spaces
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' +', ' ', text)

            return text.strip()

        except Exception as e:
            raise ValueError(f"Failed to parse content from URL '{url}': {str(e)}")

    def parse_any(self, source: str) -> dict:
        """
        Auto-detect the source type and parse content.

        Args:
            source (str): File path or web URL.

        Returns:
            dict: Parsed details containing:
                  {"text": str, "source_type": str, "char_count": int, "title": str|None}
        """
        source_cleaned = source.strip()
        source_lower = source_cleaned.lower()

        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            source_type = "url"
            self.last_title = None
            text = self.parse_url(source_cleaned)
            title = self.last_title
        elif source_lower.endswith(".pdf"):
            source_type = "pdf"
            text = self.parse_pdf(source_cleaned)
            title = None
            try:
                with fitz.open(source_cleaned) as doc:
                    title = doc.metadata.get("title")
            except Exception:
                pass
            if not title:
                title = os.path.basename(source_cleaned)
        elif source_lower.endswith(".txt"):
            source_type = "txt"
            text = self.parse_txt(source_cleaned)
            title = os.path.basename(source_cleaned)
        else:
            raise ValueError(
                f"Unsupported source format or extension: '{source}'. "
                "Only .txt, .pdf files, and http/https URLs are supported."
            )

        return {
            "text": text,
            "source_type": source_type,
            "char_count": len(text),
            "title": title
        }

    def _clean_page_layout(self, page_text: str) -> str:
        """
        Helper method to clean headers, footers, hyphens, and merged lines.
        """
        lines = [line.strip() for line in page_text.split("\n")]
        
        # Strip leading/trailing empty lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        if not lines:
            return ""

        # Remove header page number heuristics (top line)
        first_line = lines[0]
        if (first_line.isdigit() or
            re.match(r'(?i)^(page|pg\.?)\s*\d+', first_line) or
            re.match(r'(?i)^\d+\s*(of|/)\s*\d+$', first_line)):
            lines.pop(0)

        if not lines:
            return ""

        # Remove footer page number heuristics (bottom line)
        last_line = lines[-1]
        if (last_line.isdigit() or
            re.match(r'(?i)^(page|pg\.?)\s*\d+', last_line) or
            re.match(r'(?i)^\d+\s*(of|/)\s*\d+$', last_line)):
            lines.pop()

        # Reconstruct text
        text = "\n".join(lines)

        # Fix hyphenation (e.g. "deve-\nlopment" -> "development")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Merge broken lines: split by double newlines (paragraphs),
        # replace single newlines inside paragraphs with spaces.
        paragraphs = re.split(r'\n\s*\n', text)
        cleaned_paragraphs = []
        for para in paragraphs:
            para_cleaned = re.sub(r'\s*\n\s*', ' ', para)
            if para_cleaned.strip():
                cleaned_paragraphs.append(para_cleaned.strip())

        return "\n\n".join(cleaned_paragraphs)


# Module-level backward-compatible function wrapper for app.py
def parse_file(file_path: str) -> str:
    """
    Backward-compatible module wrapper that instantiates FileParser
    and parses the given file.
    """
    parser = FileParser()
    result = parser.parse_any(file_path)
    return result["text"]


if __name__ == "__main__":
    print("=== FileParser Demonstration ===")
    
    # 1. Setup temporary test file
    temp_txt_path = "demo_test_document.txt"
    try:
        with open(temp_txt_path, "w", encoding="utf-8") as f:
            f.write("Line 1 of text document.\nLine 2 of text document.\n\nParagraph 2 line 1.\nParagraph 2 line 2.")

        parser = FileParser()
        
        # Test .txt parsing
        print(f"\n--- Testing Text Ingestion from {temp_txt_path} ---")
        result = parser.parse_any(temp_txt_path)
        print("Source Type:", result["source_type"])
        print("Title:", result["title"])
        print("Characters:", result["char_count"])
        print("Content:")
        print(result["text"])
        
        # Test URL parsing
        print("\n--- Testing URL Ingestion ---")
        test_url = "https://www.example.com"
        url_result = parser.parse_any(test_url)
        print("Source Type:", url_result["source_type"])
        print("Title:", url_result["title"])
        print("Characters:", url_result["char_count"])
        print("Content:")
        print(url_result["text"][:300])

    except Exception as e:
        print("Error during demonstration:", e)
    finally:
        if os.path.exists(temp_txt_path):
            os.remove(temp_txt_path)
            
    print("\n=== Demo Complete ===")
