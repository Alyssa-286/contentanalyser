from summarizer.preprocessor import TextPreprocessor

from summarizer.extractive import ExtractiveSummarizer

p = ExtractiveSummarizer()
text = "OpenAI is an American artificial intelligence (AI) research laboratory. It consists of the non-profit OpenAI Inc. and its for-profit subsidiary corporation OpenAI Global LLC. OpenAI conducts AI research with the declared intention of promoting and developing a friendly AI. The organization was founded in San Francisco in late 2015 by Sam Altman, Reid Hoffman, Jessica Livingston, Elon Musk, Ilya Sutskever, Wojciech Zaremba, and Peter Thiel. Microsoft provides OpenAI with a $13 billion investment as of 2023."

sents = p.preprocessor.tokenize_sentences(text)
print("Number of sentences:", len(sents))
for i, s in enumerate(sents):
    print(f"  {i}: {s}")

scores = p._get_sentence_scores(sents)
print("Sentence scores:", scores)

summary = p.summarize(text, ratio=0.3)
print("Summary:")
print(summary)
