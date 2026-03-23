import re
import unidecode
import contractions


def clean_text(text):
    """Clean and normalize a piece of text for model input."""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # only run unidecode if there's actually non-ASCII content
    if any(ord(c) > 127 for c in text):
        text = unidecode.unidecode(text)

    if "'" in text:
        text = contractions.fix(text)

    # strip RT prefix
    text = re.sub(r"^rt\s+", "", text)

    # replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    # hashtags and mentions
    text = re.sub(r"#\w+", "[HASHTAG]", text)
    text = re.sub(r"@\w+", "[MENTION]", text)

    # standalone numbers (don't want to remove years inside words)
    text = re.sub(r"\b\d+\b", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_label(label):
    """Map various label formats to 0/1 int."""
    if label in [1, "1", "TRUE", "True", "true", True]:
        return 1
    return 0
