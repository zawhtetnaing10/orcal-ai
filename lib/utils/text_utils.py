import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def tokenize_text(text: str):
    """
        Create tokens for the given text.
    """

    # Change to lower case
    text = text.lower()

    # Remove punctuation
    trans_table = str.maketrans("", "", string.punctuation)
    text = text.translate(trans_table)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    filtered_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            filtered_tokens.append(token)

    # Stem
    result = [STEMMER.stem(token) for token in filtered_tokens]

    return result
