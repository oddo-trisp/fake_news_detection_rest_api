import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.models.SupervisedLearner import SupervisedLearner
from src.utils.conf import *


class FakeNewsClassifier(SupervisedLearner):

    def __init__(self, _model_name, _df_train=None, _df_test=None):
        super().__init__(_model_name, _df_train, _df_test)

    def prepare_data(self, df_train=None, df_test=None):
        if df_train is None:
            df_train = pd.read_csv(TRAIN_PATH)
        if df_test is None:
            df_test = pd.read_csv(TEST_PATH)

        df_train = FakeNewsClassifier.engineer_data(df_train)
        self.X_train = df_train['total'].values
        self.y_train = df_train['label'].values

        self.test_id = df_test['id']
        self.prepare_test_data(df_test)
        self.X_test_clear = self.X_test

    def prepare_test_data(self, df_test):
        df_test['label'] = 't'
        df_test = FakeNewsClassifier.engineer_data(df_test, remove_outliers=False)
        self.X_test = df_test['total'].values

    def create_pipeline(self):

        if self.model_name is EXTRA_TREES:
            pipeline = Pipeline([
                ('bow', CountVectorizer(ngram_range=(1, 2))),
                ('clf', ExtraTreesClassifier(n_estimators=5,n_jobs=4))
            ])
        elif self.model_name is ADA_BOOST:
            pipeline = Pipeline([
                ('bow', CountVectorizer(ngram_range=(1, 2))),
                ('clf', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5))
            ])
        elif self.model_name is RANDOM_FOREST:
            pipeline = Pipeline([
                ('bow', CountVectorizer(ngram_range=(1, 2))),
                ('clf', RandomForestClassifier(n_estimators=5))
            ])
        elif self.model_name is MULTINOMIAL_NB:
            pipeline = Pipeline([
                ('bow', CountVectorizer(ngram_range=(1, 2))),
                ('clf', MultinomialNB())
            ])
        elif self.model_name is LOGISTIC_REGRESSION:
            pipeline = Pipeline([
                ('bow', CountVectorizer(ngram_range=(1, 2))),
                ('clf', LogisticRegression())
            ])
        else:
            pipeline = Pipeline([
                ('bow', CountVectorizer(ngram_range=(1, 2))),
                ('clf', LogisticRegression())
            ])

        return pipeline

    @staticmethod
    def engineer_data(data_change, remove_outliers=True):
        data_change = FakeNewsClassifier.remove_useless_columns(data_change)

        if remove_outliers is True:
            data_change = FakeNewsClassifier.fill_na_values(data_change, columns=['title'])
            data_change = FakeNewsClassifier.remove_na_values(data_change)
            data_change = FakeNewsClassifier.remove_outliers(data_change)
        else:
            data_change = FakeNewsClassifier.fill_na_values(data_change)

        data_change = FakeNewsClassifier.create_features(data_change)

        return data_change
