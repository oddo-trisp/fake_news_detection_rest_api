import re

from nltk.stem.snowball import PorterStemmer
from sklearn.base import TransformerMixin, BaseEstimator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

import src.utils.w2v as w2v


class AverageWordVectorTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, language='english', stop_words=None, vocabulary_data=None):
        self.language = language
        self.stop_words = stop_words
        self.vocabulary_data = vocabulary_data
        self.w2v_model = w2v.get_w2v_model(data=vocabulary_data, language=language)
        self.vocabulary_size, self.embedding_size = self.w2v_model.wv.vectors.shape

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _X = w2v.prepare_w2v_data(data=X, language=self.language, stop_words=self.stop_words)
        return _X


class PadSequencesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=5000, max_length=20, stopwords=None):
        self.vocabulary_size = vocabulary_size
        self.max_length = max_length
        self.stopwords = stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        messages = X.copy()

        ps = PorterStemmer()
        corpus_train = []
        for i in range(len(messages)):
            review = re.sub('[^a-zA-Z]', ' ', messages[i])
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if word not in self.stopwords]
            review = ' '.join(review)
            corpus_train.append(review)

        one_hot_rep = [one_hot(words, self.vocabulary_size) for words in corpus_train]
        _X = pad_sequences(one_hot_rep, padding='pre', maxlen=self.max_length)
        return _X


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.todense()
