import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.SupervisedLearner import SupervisedLearner
from src.models.transformers import AverageWordVectorTransformer
from src.utils.conf import *


class FakeNewsClassifier(SupervisedLearner):

    def __init__(self, learner_name, feature_name, evaluate, language=ENGLISH, df_train=None, df_test=None):
        super().__init__(learner_name, feature_name, evaluate, language, df_train, df_test)

    # TODO: Add optimal parameters from grid search
    def create_learner(self):
        clf = None

        if self.learner_name is EXTRA_TREES:
            clf = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, n_jobs=4)
        elif self.learner_name is ADA_BOOST:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=N_ESTIMATORS)
        elif self.learner_name is GAUSSIAN_NB:
            clf = GaussianNB()
        elif self.learner_name is LOGISTIC_REGRESSION:
            clf = LogisticRegression()
        elif self.learner_name is RANDOM_FOREST:
            clf = RandomForestClassifier(n_estimators=N_ESTIMATORS)
        elif self.learner_name is SVM:
            clf = SVC(kernel='linear', probability=True, cache_size=1000)

        return {'clf': clf}

    def create_default_learner(self):

        # Create default classifier
        clf = None
        parameters = {}

        # TODO: Add parameters
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
            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            parameters = {'n_estimators': n_estimators, 'max_features': max_features,
                          'max_depth': max_depth, 'min_samples_split': min_samples_split,
                          'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
        elif self.learner_name is SVM:
            clf = SVC()

        return clf, parameters

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
