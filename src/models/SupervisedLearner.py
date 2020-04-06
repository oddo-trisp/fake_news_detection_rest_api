import datetime
import pickle
from time import time
from os import path

import pandas as pd
import numpy as np

from src.models.ISupervisedLearner import ISupervisedLearner
from src.utils.conf import *


class SupervisedLearner(ISupervisedLearner):

    def __init__(self, _model_name, _df_train=None, _df_test=None):
        super().__init__(_model_name, _df_train, _df_test)

        self.test_id = None

        self.X_train = None
        self.y_train = None

        self.X_test_clear = None
        self.X_test = None

        self.model = None
        self.model_name = _model_name

        self.prepare_data(_df_train, _df_test)

        if _model_name is not None:
            self.init_model()

    def prepare_data(self, df_train=None, df_test=None):
        pass

    def prepare_test_data(self, df_test):
        pass

    def fit(self, _model):
        start_time = time()

        self.model = _model
        self.model.fit(self.X_train, self.y_train)

        print("Time to Fit ({}): {:.4f}s ".format(self.model_name, time() - start_time))
        print("Score: {:.4f} \n".format(self.model.score(self.X_train, self.y_train)))

        pickle.dump(self.model, open(PIPELINE_PATH + self.model_name + FORMAT_SAV, 'wb'))

    def predict_proba(self, _df_test=None):
        if _df_test is not None:
            self.prepare_test_data(_df_test)

        start_time = time()

        y_test = self.model.predict_proba(self.X_test)
        y_test = np.asanyarray([['%.5f' % elem for elem in subarray] for subarray in y_test])
        y_test = pd.DataFrame(y_test, columns=['0', '1'])

        self.prediction_to_csv(self.model_name, self.test_id, y_test)
        print("Time to Predict ({}): {:.4f}s \n".format(self.model_name, time() - start_time))

        result = y_test['1'].values[0]
        return result

    def init_model(self):
        full_model_name = PIPELINE_PATH + self.model_name + FORMAT_SAV
        if path.exists(full_model_name):
            self.load_pipeline(full_model_name)
        else:
            self.fit(self.create_pipeline())

    def load_pipeline(self, name):
        self.model = pickle.load(open(name, 'rb'))

    def create_pipeline(self):
        pass

    @staticmethod
    def prediction_to_csv(filename, test_id, y_test):
        sub = pd.DataFrame(y_test, columns=['0', '1'])
        sub['id'] = test_id
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        sub.to_csv(OUTPUT_PATH + filename + "_" + now + FORMAT_CSV, index=False, sep=',')

    @staticmethod
    def engineer_data(data_change, remove_outliers=True):
        pass

    @staticmethod
    def remove_outliers(data):
        indexes = data[data['text'].map(len) < 50].index
        for index in indexes:
            data.drop(index, axis=0, inplace=True)
        return data

    @staticmethod
    def remove_useless_columns(data):
        cols = ('id', 'author')
        for col in cols:
            if col in data.columns:
                data.drop([col], axis=1, inplace=True)
        return data

    @staticmethod
    def fill_na_values(data, na_value=' ', columns=None):
        if columns is None:
            data = data.fillna(value=na_value)
        else:
            data[columns] = data[columns].fillna(value=na_value)
        return data

    @staticmethod
    def remove_na_values(data):
        data = data.dropna()
        return data

    @staticmethod
    def create_features(data):
        data['total'] = data['title'] + ' ' + data['text']
        return data
