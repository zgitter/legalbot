import os
import json
import nltk
import pickle
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Ensure NLTK tokenizer is available
nltk.download("punkt")

# Load JSON law book
with open("data/Ethiopia_1994.json", "r", encoding="utf-8") as f:
    law_data = json.load(f)

def process_section(section):
    """Extract text content from section (handles dict, list, and nested structures)."""
    if isinstance(section, dict):
        return str(section.get("content", "")) + "\n" + process_section(section.get("section", ""))
    elif isinstance(section, list):
        return "\n".join(process_section(subsec) for subsec in section)
    return str(section)  # Ensure all values are strings

def process_chapter(chapter):
    """Extract articles from a chapter and return a list of article texts."""
    documents = []
    for article in chapter.get("section", []):
        article_text = process_section(article.get("section", ""))
        documents.append(article_text)
    return documents

# Prepare documents for BM25
corpus_texts = []
for chapter in law_data:
    corpus_texts.extend(process_chapter(chapter))

# Tokenize documents for BM25
tokenized_corpus = [word_tokenize(text.lower()) for text in corpus_texts]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Save BM25 index to file
with open("data/bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, corpus_texts), f)

print("Successfully created and saved BM25 index!")