import re
import logging
from collections import Counter

try:
    import spacy
    _spacy_ok = True
except ImportError:
    _spacy_ok = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    _nltk_ok = True
except ImportError:
    _nltk_ok = False

# lazy-loaded globals
nlp = None
sia = None

IMPORTANT_FEATURES = [
    "sentiment_polarity",
    "char_count",
    "avg_word_length",
    "verb_ratio",
    "word_count",
    "type_token_ratio",
    "readability_score",
]


def _init_models():
    global nlp, sia

    if _spacy_ok and nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        except OSError:
            print("spaCy model not found, run: python -m spacy download en_core_web_sm")

    if _nltk_ok and sia is None:
        try:
            sia = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"VADER init failed: {e}")


def extract_basic_features(text):
    features = {}
    features["char_count"] = len(text)

    words = text.split()
    features["word_count"] = len(words)

    sentences = re.split(r"[.!?]+", text)
    features["sentence_count"] = max(1, len([s for s in sentences if s.strip()]))

    features["avg_word_length"] = round(len(text) / max(1, len(words)), 3)
    features["type_token_ratio"] = round(len(set(words)) / max(1, len(words)), 3)
    features["readability_score"] = round(len(words) / max(1, features["sentence_count"]), 3)

    return features


def extract_nlp_features(text):
    global nlp
    features = {
        "noun_ratio": 0.0,
        "verb_ratio": 0.0,
        "adj_ratio": 0.0,
        "adv_ratio": 0.0,
        "num_named_entities": 0,
    }

    if nlp is None:
        _init_models()
    if nlp is None:
        return features

    # truncate long texts so spacy doesn't OOM
    doc = nlp(text[:5000])
    total = max(1, len(doc))
    pos_counts = Counter(t.pos_ for t in doc)

    features["noun_ratio"] = round(pos_counts.get("NOUN", 0) / total, 3)
    features["verb_ratio"] = round(pos_counts.get("VERB", 0) / total, 3)
    features["adj_ratio"] = round(pos_counts.get("ADJ", 0) / total, 3)
    features["adv_ratio"] = round(pos_counts.get("ADV", 0) / total, 3)
    features["num_named_entities"] = len(doc.ents)

    return features


def extract_sentiment(text):
    global sia
    if sia is None:
        _init_models()

    if sia is not None:
        scores = sia.polarity_scores(text)
        return {
            "sentiment_polarity": round(scores["compound"], 3),
            "sentiment_positive": round(scores["pos"], 3),
            "sentiment_negative": round(scores["neg"], 3),
            "sentiment_neutral": round(scores["neu"], 3),
        }

    # rough fallback if VADER not available
    pos_words = {"good", "great", "positive", "support", "help", "safe"}
    neg_words = {"bad", "crime", "danger", "attack", "illegal", "threat"}
    words = set(text.lower().split())
    pos = sum(1 for w in pos_words if w in words)
    neg = sum(1 for w in neg_words if w in words)
    total = max(1, len(words))
    return {
        "sentiment_positive": round(pos / total, 3),
        "sentiment_negative": round(neg / total, 3),
        "sentiment_neutral": round(1 - pos / total - neg / total, 3),
        "sentiment_polarity": round((pos - neg) / total, 3),
    }


def extract_all_features(text):
    if not text:
        return {}
    features = extract_basic_features(text)
    features.update(extract_nlp_features(text))
    features.update(extract_sentiment(text))
    return features
