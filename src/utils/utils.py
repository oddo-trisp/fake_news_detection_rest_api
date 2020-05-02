# Utility methods

import os
import nltk
from nltk.corpus import stopwords
from pathlib import Path


def get_valid_path(destination):
    root_path = Path(__file__).parent.parent.parent
    working_dir = os.getcwd()
    steps = Path(working_dir).relative_to(root_path).as_posix().count('/') + 1

    prefix = ''
    for i in range(steps):
        prefix = prefix + '../'

    return Path(prefix + destination)


def get_tokenizer():
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return tokenizer


def get_stopwords():
    try:
        stop_words = stopwords.words("english")
    except LookupError:
        nltk.download('stopwords')
        stop_words = stopwords.words("english")

    return set(stop_words)
