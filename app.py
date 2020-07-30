import warnings

import newspaper
from flask import Flask, request, abort
from flask import jsonify
from flask_cors import CORS
from newspaper import Article
from pandas import json_normalize

import src.utils.utils as utils
from src.models.FakeNewsClassifier import FakeNewsClassifier
from src.models.FakeNewsDeepLearner import FakeNewsDeepLearner
from src.utils.conf import *

# def ignore_warn():
#    pass

warnings.filterwarnings('ignore')


# warnings.warn = ignore_warn


class FakeNewsDetector(Flask):
    def __init__(self, *args, **kwargs):
        super(FakeNewsDetector, self).__init__(*args, **kwargs)

        model_name = RANDOM_FOREST
        feature_name = W2V

        self.df_train = utils.read_csv(utils.get_valid_path(TRAIN_PATH))
        self.df_test = utils.read_csv(utils.get_valid_path(TEST_PATH))
        self.fake_news_learner = FakeNewsClassifier(model_name, feature_name, False, ENGLISH, self.df_train,
                                                    self.df_test) \
            if model_name in CLASSIFICATION_SET \
            else FakeNewsDeepLearner(model_name, self.df_train, False, ENGLISH, self.df_test)

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
CORS(app, resources={r"/*": {"origins": "*"}})  # TODO: add the domain of production


@app.route('/test')
def test():
    y_test = app.fake_news_learner.predict_proba()
    return {'probability': 0.00}, 200


@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'title' not in request.json or 'text' not in request.json:
        abort(400)

    article = json_normalize(request.json)
    if article.shape[0] is not 1:
        abort(400)

    result = app.fake_news_learner.predict_proba(article)

    return {'probability': result}, 200


@app.route('/scraper', methods=['POST'])
def scraper():
    if not request.json or 'url' not in request.json:
        abort(400)

    # The Basics of downloading the article to memory
    article = Article(request.json['url'])
    article.download()
    article.parse()
    article.nlp()

    return {'title': article.title, 'text': article.text}, 200

if __name__ == '__main__':
    app.run()
