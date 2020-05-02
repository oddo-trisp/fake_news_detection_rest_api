import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.SupervisedLearner import SupervisedLearner
from src.models.w2v import *
from src.utils.conf import *
from src.utils.utils import *


class FakeNewsClassifier(SupervisedLearner):

    def __init__(self, _learner_name, _feature_name, _evaluate, _df_test=None, _df_train=None):
        super().__init__(_learner_name, _feature_name, _evaluate, _df_test, _df_train)

    def prepare_data(self, df_train=None, df_test=None):
        if df_train is None:
            df_train = pd.read_csv(get_valid_path(TRAIN_PATH))
        if df_test is None:
            df_test = pd.read_csv(get_valid_path(TEST_PATH))

        self.prepare_train_data(df_train)

        self.prepare_test_data(df_test)

    def prepare_train_data(self, df_train):
        df_train = self.engineer_data(df_train)

        if self.feature_name is W2V:
            self.X_train = w2v_prepare(df_train)
        else:
            self.X_train = df_train['total'].values

        self.y_train = df_train['label'].values

    def prepare_test_data(self, df_test):
        self.test_id = df_test['id']
        df_test['label'] = 't'
        df_test = self.engineer_data(df_test, remove_outliers=False)

        if self.feature_name is W2V:
            self.X_test = w2v_prepare(df_test)
        else:
            self.X_test = df_test['total'].values

        # self.X_test = df_test['total'].values
        # self.X_test_clear = self.X_test

    def create_pipeline(self, learner=None, features=None):

        classifier = self.create_learner() if learner is None else learner
        features = self.create_features() if features is None else features

        steps = []
        for k, v in features.items():
            if v is not None:
                steps.append((k, v))

        for k, v in classifier.items():
            if v is not None:
                steps.append((k, v))

        pipeline = Pipeline(steps=steps)

        return pipeline

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
        bow = None
        lsa = None

        if self.feature_name is BOW:
            bow = CountVectorizer(ngram_range=N_GRAM_RANGE, max_features=MAX_FEATURES)
        elif self.feature_name is SVD:
            bow = CountVectorizer(ngram_range=N_GRAM_RANGE, max_features=MAX_FEATURES)

            n_samples, n_components = TfidfVectorizer(max_features=MAX_FEATURES).fit_transform(self.X_train,
                                                                                               self.y_train).shape
            n_components = int(n_components * 0.9)  # 90% components
            lsa = TruncatedSVD(n_components=n_components)

        return {'bow': bow, 'lsa': lsa}

    @staticmethod
    def engineer_data(data, remove_outliers=True):
        data = FakeNewsClassifier.remove_useless_columns(data)

        if remove_outliers is True:
            data = FakeNewsClassifier.fill_na_values(data, columns=['title'])
            data = FakeNewsClassifier.remove_na_values(data)
            data = FakeNewsClassifier.remove_outliers(data)
        else:
            data = FakeNewsClassifier.fill_na_values(data)

        data = FakeNewsClassifier.create_new_columns(data)

        return data
