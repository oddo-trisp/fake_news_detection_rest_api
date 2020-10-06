import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

from src.models.SupervisedLearner import SupervisedLearner
from src.models.transformers import AverageWordVectorTransformer, DenseTransformer
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

        if self.learner_name is NEAREST_CENTROID:  # Rocchio Algorithm
            clf = NearestCentroid()
        elif self.learner_name is ADA_BOOST:  # Boosting
            clf = AdaBoostClassifier()
        elif self.learner_name is LOGISTIC_REGRESSION:  # Logistic Regression
            clf = LogisticRegression()
        elif self.learner_name is GAUSSIAN_NB:  # Naive Bayes
            clf = GaussianNB()
        elif self.learner_name is KNN:  # K-nearest Neighbor (Bagging)
            clf = KNeighborsClassifier()
        elif self.learner_name is SVM:  # Support Vector Machine (SVM)
            clf = SVC(kernel='linear', probability=True)
        elif self.learner_name is EXTRA_TREES:  # Decision Tree
            clf = ExtraTreesClassifier()
        elif self.learner_name is RANDOM_FOREST:  # Random Forest
            clf = RandomForestClassifier()

        return {CLF: clf}

    def create_features(self):
        vect = None
        tfidf = None
        dense = None
        svd = None

        stop_words = self.get_stopwords()

        if self.feature_name is BOW:
            vect = CountVectorizer(ngram_range=N_GRAM_RANGE, max_features=MAX_FEATURES, stop_words=stop_words)
            if self.learner_name in DENSE_SET:
                dense = DenseTransformer()
        elif self.feature_name is TF_IDF:
            vect = CountVectorizer(ngram_range=N_GRAM_RANGE, max_features=MAX_FEATURES, stop_words=stop_words)
            tfidf = TfidfTransformer(smooth_idf=False)
            if self.learner_name in DENSE_SET:
                dense = DenseTransformer()
        elif self.feature_name is TRUNC_SVD:
            vect = CountVectorizer(ngram_range=N_GRAM_RANGE, max_features=MAX_FEATURES, stop_words=stop_words)
            tfidf = TfidfTransformer(smooth_idf=False)

            n_samples, n_components = TfidfVectorizer(max_features=MAX_FEATURES, stop_words=stop_words).fit_transform(
                self.X_train, self.y_train).shape
            n_components = int(n_components * 0.1)  # 50% components
            svd = TruncatedSVD(n_components=n_components)
        elif self.feature_name is W2V:
            vect = AverageWordVectorTransformer(language=self.language, stop_words=stop_words,
                                                vocabulary_data=self.X_train)

        return {VECT: vect, TFIDF: tfidf, DENSE: dense, SVD: svd}

    def get_evaluation_params(self):

        parameters = {}

        if self.learner_name is NEAREST_CENTROID:  # Rocchio Algorithm
            metric = ['euclidean', 'manhattan', 'minkowski', 'cosine']
            shrinkage = [None, .2]
            parameters = {'metric': metric, 'shrink_threshold': shrinkage}
        elif self.learner_name is ADA_BOOST:  # Boosting
            n_estimators = [50, 100, 500]
            learning_rate = [0.001, 0.01, .1, 1.]
            parameters = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
        elif self.learner_name is LOGISTIC_REGRESSION:  # Logistic Regression
            C = [0.0001, 0.01, 0.05, 0.2, 1]
            solver = ['newton-cg', 'lbfgs', 'liblinear']
            parameters = {"C": C, "solver": solver}
        elif self.learner_name is GAUSSIAN_NB:  # Naive Bayes
            var_smoothing = np.logspace(0, -9, num=100)
            parameters = {'var_smoothing': var_smoothing}
        elif self.learner_name is KNN:  # K-nearest Neighbor (Bagging)
            n_neighbors = range(1, 6, 2)
            weights = ['uniform', 'distance']
            metric = ['euclidean', 'manhattan', 'minkowski']
            p = [2, 5]
            parameters = {'n_neighbors': n_neighbors, 'weights': weights,
                          'metric': metric, 'p': p}
        elif self.learner_name is SVM:  # Support Vector Machine (SVM)
            C = [0.1, 1, 10, 100]
            gamma = [1, 0.1, 0.01, 0.001]
            parameters = {'C': C, 'gamma': gamma}
        elif self.learner_name is EXTRA_TREES \
                or self.learner_name is RANDOM_FOREST:  # Decision Tree or Random Forest
            n_estimators = [100, 1000]
            max_features = ['auto', 'sqrt']
            max_depth = [10, 100]
            min_samples_split = [5, 10]
            min_samples_leaf = [2, 4]
            parameters = {'n_estimators': n_estimators, 'max_features': max_features,
                          'max_depth': max_depth, 'min_samples_split': min_samples_split,
                          'min_samples_leaf': min_samples_leaf}

        return self.add_clf_prefix(params=parameters)

    @staticmethod
    def get_fit_params():
        return {}
