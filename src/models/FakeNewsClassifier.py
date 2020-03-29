import datetime
import pickle
from time import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from src.models.IClassifier import IClassifier
from src.utils.conf import *


class FakeNewsClassifier(IClassifier):

    def __init__(self, _df_train=None, _df_test=None, _model=None, _transformation=None, _vectorizer=None):
        super().__init__()

        self.test_id = None

        self.X_train = None
        self.y_train = None

        self.X_test_clear = None
        self.X_test = None

        self.model = None

        if _transformation is None:
            self.transformation = TfidfTransformer(smooth_idf=False)
        else:
            self.transformation = _transformation

        if _vectorizer is None:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        else:
            self.vectorizer = _vectorizer

        self.prepare_data(_df_train, _df_test)

        if _model is not None:
            self.fit(_model)

    def prepare_data(self, df_train=None, df_test=None):
        if df_train is None:
            df_train = pd.read_csv(TRAIN_PATH)
        if df_test is None:
            df_test = pd.read_csv(TEST_PATH)

        df_train = FakeNewsClassifier.engineer_data(df_train)
        # df_train_transformed, df_train_vectroized = \
        #     FakeNewsClassifier.create_transformations(df_train['total'].values, self.transformation, self.vectorizer)
        # self.X_train = df_train_transformed
        # self.y_train = df_train['total'].values
        self.X_train = df_train['total'].values
        self.y_train = df_train['label'].values

        self.test_id = df_test['id']
        self.prepare_test_data(df_test)
        self.X_test_clear = self.X_test

    def prepare_test_data(self, df_test):
        df_test['label'] = 't'
        df_test = FakeNewsClassifier.engineer_data(df_test)
        # df_test_transformed, df_test_vectroized = \
        #    FakeNewsClassifier.create_transformations(df_test['total'].values, self.transformation, self.vectorizer)
        # self.X_test = df_test_transformed
        self.X_test = df_test['total'].values

    def fit(self, _model):
        start_time = time()

        self.model = _model
        self.model.fit(self.X_train, self.y_train)

        print("Time to Fit ({}): {:.4f}s \n".format(self.model['clf'].__class__.__name__, time() - start_time))
        print("Score: {:.4f} \n".format(self.model.score(self.X_train, self.y_train)))

        pickle.dump(self.model, open(MODEL_PATH, 'wb'))

    def predict_proba(self, _df_test=None):
        if _df_test is not None:
            self.prepare_test_data(_df_test)

        start_time = time()

        y_test = self.model.predict_proba(self.X_test)
        y_test = np.asanyarray([['%.5f' % elem for elem in subarray] for subarray in y_test])
        y_test = pd.DataFrame(y_test, columns=['0', '1'])

        model_name = self.model['clf'].__class__.__name__
        # self.prediction_to_csv(model_name, self.test_id, y_test)
        print("Time to Predict ({}): {:.4f}s \n".format(model_name, time() - start_time))

        result = y_test['1'].values[0]
        return result

    def load_model(self):
        self.model = pickle.load(open(MODEL_PATH, 'rb'))

    @staticmethod
    def prediction_to_csv(filename, test_id, y_test):
        sub = pd.DataFrame(y_test, columns=['0', '1'])
        sub['id'] = test_id
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        sub.to_csv(OUTPUT_PATH + filename + "_" + now + FORMAT_CSV, index=False, sep=',')

    @staticmethod
    def engineer_data(data_change):
        data_change = FakeNewsClassifier.drop_useless_columns(data_change)
        data_change = FakeNewsClassifier.fill_na_values(data_change)
        data_change = FakeNewsClassifier.create_features(data_change)

        return data_change

    @staticmethod
    def remove_outliers(data):
        pass

    @staticmethod
    def drop_useless_columns(data):
        cols = ('id', 'author')
        for col in cols:
            if col in data.columns:
                data.drop([col], axis=1, inplace=True)
        return data

    @staticmethod
    def fill_na_values(data, na_value=' '):
        data = data.fillna(na_value)
        return data

    @staticmethod
    def create_features(data):
        data['total'] = data['title'] + ' ' + data['text']
        return data

    @staticmethod
    def create_transformations(data, transformation=None, vectorizer=None):
        vectorized_data = vectorizer.fit_transform(data)
        transformed_data = transformation.fit_transform(vectorized_data)
        return transformed_data, vectorized_data
