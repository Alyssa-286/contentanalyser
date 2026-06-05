import sys
import traceback

try:
    from summarizer.extractive import ExtractiveSummarizer
    print("IMPORT OK")
except Exception as e:
    print("IMPORT FAILED:", type(e).__name__, str(e))
    traceback.print_exc()

try:
    from utils.text_cleaner import clean_sentence, is_noise_sentence, clean_text
    print("TEXT CLEANER OK")
except Exception as e:
    print("TEXT CLEANER FAILED:", type(e).__name__, str(e))
    traceback.print_exc()
