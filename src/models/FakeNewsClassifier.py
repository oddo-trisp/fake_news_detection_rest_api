import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from src.models.SupervisedLearner import SupervisedLearner
from src.models.transformers import AverageWordVectorTransformer
from src.utils.conf import *


class FakeNewsClassifier(SupervisedLearner):

    def __init__(self, learner_name, feature_name, evaluate, language=ENGLISH, df_train=None, df_test=None):
        self.validate_init(learner_name, feature_name)
        super().__init__(learner_name, feature_name, evaluate, language, df_train, df_test)

    def validate_init(self, learner_name, feature_name):
        super().validate_init(learner_name, feature_name)
        if learner_name not in CLASSIFICATION_SET or feature_name not in CLASSIFICATION_FEATURE_SET:
            raise Exception('learner_name and feature_name must belong to classification set')

    def create_learner(self):
        clf = None

        if self.learner_name is EXTRA_TREES:
            clf = ExtraTreesClassifier()
        elif self.learner_name is ADA_BOOST:
            clf = AdaBoostClassifier()
        elif self.learner_name is GAUSSIAN_NB:
            clf = GaussianNB()
        elif self.learner_name is LOGISTIC_REGRESSION:
            clf = LogisticRegression()
        elif self.learner_name is RANDOM_FOREST:
            clf = RandomForestClassifier()
        elif self.learner_name is SVM:
            clf = SVC(kernel='linear', probability=True)

        return {'clf': clf}

    def create_features(self):
        vect = None
        tfidf = None
        svd = None

        stop_words = self.get_stopwords()

        if self.feature_name is BOW:
            vect = CountVectorizer(ngram_range=N_GRAM_RANGE, max_features=MAX_FEATURES, stop_words=stop_words)
            tfidf = TfidfTransformer(smooth_idf=False)
        elif self.feature_name is SVD:
            vect = CountVectorizer(ngram_range=N_GRAM_RANGE, max_features=MAX_FEATURES, stop_words=stop_words)
            tfidf = TfidfTransformer(smooth_idf=False)

            n_samples, n_components = TfidfVectorizer(max_features=MAX_FEATURES, stop_words=stop_words).fit_transform(
                self.X_train, self.y_train).shape
            n_components = int(n_components * 0.9)  # 90% components
            svd = TruncatedSVD(n_components=n_components)
        elif self.feature_name is W2V:
            vect = AverageWordVectorTransformer(language=self.language, stop_words=stop_words,
                                                vocabulary_data=self.X_train)

        return {'vect': vect, 'tfidf': tfidf, 'svd': svd}

    def get_evaluation_params(self):

        parameters = {}

        # TODO: Add parameters
        if self.learner_name is EXTRA_TREES:
            parameters = {}
        elif self.learner_name is ADA_BOOST:
            parameters = {}
        elif self.learner_name is GAUSSIAN_NB:
            parameters = {}
        elif self.learner_name is LOGISTIC_REGRESSION:
            parameters = {}
        elif self.learner_name is RANDOM_FOREST:
            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            parameters = {'clf__n_estimators': n_estimators, 'clf__max_features': max_features,
                          'clf__max_depth': max_depth, 'clf__min_samples_split': min_samples_split,
                          'clf__min_samples_leaf': min_samples_leaf, 'clf__bootstrap': bootstrap}
        elif self.learner_name is SVM:
            parameters = {}

        return parameters

    @staticmethod
    def get_fit_params():
        return None

