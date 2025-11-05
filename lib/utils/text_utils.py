import nltk
from nltk.tokenize import word_tokenize


def tokenize_text(text):
    """
        Create tokens for the given text.
    """
    return word_tokenize(text.lower())
