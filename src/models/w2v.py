import pickle
import re
from os import path

import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from src.utils.conf import *


# TODO check if need to add model for test set
def w2v_prepare(data):
    full_model_name = PIPELINE_PATH + 'Word2Vec' + FORMAT_SAV
    if path.exists(full_model_name):
        with open(full_model_name, 'rb') as f:
            model = pickle.load(f)
    else:
        model = create_w2v_model(data)

    clean_train_reviews = []
    for review in data['total']:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    return get_avg_feature_vectors(clean_train_reviews, model)


def create_w2v_model(data):
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        nltk.download('stopwords')

    sentences = []
    for content in data['total']:
        sentences += review_to_sentences(content, tokenizer)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10
    downsampling = 1e-3

    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                     window=context,
                     sample=downsampling)
    model.init_sims(replace=True)

    with open(PIPELINE_PATH + 'Word2Vec' + FORMAT_SAV, 'wb') as f:
        pickle.dump(model, f)

    return model


def review_to_wordlist(review, remove_stopwords=False):
    # Remove non-letters
    review_text = re.sub('[^a-zA-Z]', ' ', review)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def get_avg_feature_vectors(reviews, model):
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    dim = len(next(iter(w2v.values())))
    return np.array([
        np.mean([w2v[w] for w in words if w in w2v]
                or [np.zeros(dim)], axis=0)
        for words in reviews
    ])
