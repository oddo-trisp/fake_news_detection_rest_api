import warnings
from os import path

import pandas as pd
from flask import Flask, request, abort
from flask import jsonify
from pandas import json_normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.FakeNewsClassifier import FakeNewsClassifier
from src.utils.conf import *

# def ignore_warn():
#    pass


warnings.filterwarnings("ignore")


# warnings.warn = ignore_warn


class FakeNewsDetector(Flask):
    def __init__(self, *args, **kwargs):
        super(FakeNewsDetector, self).__init__(*args, **kwargs)

        self.df_train = pd.read_csv(TRAIN_PATH)
        self.df_test = pd.read_csv(TEST_PATH)

        self.fake_news_classifier = FakeNewsClassifier(self.df_train, self.df_test)

        if path.exists(MODEL_PATH):
            self.fake_news_classifier.load_model()
        else:
            pipeline = Pipeline([
                ('bow', CountVectorizer(ngram_range=(1, 2))),
                ('clf', LogisticRegression())
            ])
            self.fake_news_classifier.fit(_model=pipeline)

    def make_response(self, rv):
        # Turn the rv into a full response.
        #
        # This allows for the rv to be a dictionary which
        # indicates that the response should be JSON.
        #
        if isinstance(rv, dict):
            new_rv = jsonify(rv)
        elif isinstance(rv, tuple) and isinstance(rv[0], dict):
            new_rv = (jsonify(rv[0]), *rv[1:])
        else:
            new_rv = rv

        return super().make_response(new_rv)


app = FakeNewsDetector(__name__)


# @app.route('/test')
# def test():
#     y_test = app.fake_news_classifier.predict_proba()
#     return {'probability': app.probability}, 200

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'title' not in request.json or 'text' not in request.json:
        abort(400)

    article = json_normalize(request.json)
    if article.shape[0] is not 1:
        abort(400)

    result = app.fake_news_classifier.predict_proba(article)

    return {'probability': result}, 200


if __name__ == '__main__':
    app.run()
