import regex as re
from os import path

import nltk
import numpy as np
from gensim.models import Word2Vec

import src.utils.utils as utils
from src.utils.conf import *

full_model_name = utils.get_valid_path(MODELS_PATH + W2V_MODEL + FORMAT_SAV)


# TODO check if need to add model for test set
def prepare_w2v_data(data, language, stop_words=None, update=False):
    model = get_w2v_model(data, language, update)

    clean_reviews = []
    for review in data:
        clean_reviews.append(review_to_wordlist(review, stop_words=stop_words))
    return get_avg_feature_vectors(clean_reviews, model)


def get_w2v_model(data=None, language=None, update=False):
    if path.exists(full_model_name):
        model = load_w2v_model(data, language, update)
    else:
        model = create_w2v_model(data, language)

    return model


def create_w2v_model(data, language):
    tokenizer = get_tokenizer(language)

    sentences = []
    for content in data:
        sentences += review_to_sentences(content, tokenizer)

    # Set values for various parameters
    num_features = W2V_NUM_FEATURES  # Word vector dimensionality
    min_word_count = 50  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10
    downsampling = 1e-3

    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                     window=context,
                     sample=downsampling)
    model.init_sims(replace=True)

    utils.save_pickle_file(full_model_name, model)

    return model


def load_w2v_model(data, language, update=False):
    model = utils.load_pickle_file(full_model_name)

    if update is True:
        tokenizer = get_tokenizer(language)

        sentences = []
        for content in data:
            sentences += review_to_sentences(content, tokenizer)

        # Set values for various parameters
        total_examples = len(sentences)
        epoch = 1

        # Update model
        model.build_vocab(sentences, update=True)
        model.train(sentences, total_examples=total_examples, epochs=epoch)
        model.init_sims(replace=True)

        utils.save_pickle_file(full_model_name, model)

    return model


def review_to_wordlist(review, stop_words=None):
    # Remove non-letters
    review_text = re.sub('[^\p{L}]', ' ', review)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Optionally remove stop words (None by default)
    if stop_words is not None:
        words = [w for w in words if w not in stop_words]
    return words


def review_to_sentences(review, tokenizer, stop_words=None):
    # Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, stop_words=stop_words))
    return sentences


def get_avg_feature_vectors(reviews, model):
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    dim = len(next(iter(w2v.values())))
    avg_vectors = np.array([
        np.mean([w2v[w] for w in words if w in w2v]
                or [np.zeros(dim)], axis=0)
        for words in reviews
    ])
    return avg_vectors


def get_model_weights():
    model = get_w2v_model()
    w2v_weights = model.wv.vectors
    vocab_size, embedding_size = w2v_weights.shape

    return w2v_weights, vocab_size, embedding_size


def get_tokenizer(language):
    tokenizer_path = 'tokenizers/punkt/' + language + '.pickle'

    try:
        tokenizer = nltk.data.load(tokenizer_path)
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load(tokenizer_path)

    return tokenizer
