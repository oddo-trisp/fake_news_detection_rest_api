import datetime
import pickle
from os import path
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier

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

        self.metrics = None
        self.model = None
        self.model_name = _model_name

        self.prepare_data(_df_train, _df_test)

        if _model_name is not None:
            self.init_model()

    def prepare_data(self, df_train=None, df_test=None):
        pass

    def prepare_test_data(self, df_test):
        pass

    def train_model(self, _model):
        t1 = time()

        print("\nRunning 10-Fold test for: " + str(self.model_name))  # running prompt explaining which algorithm runs

        k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=434)  # a KFold variation
        scores = ['accuracy', 'precision_micro', 'recall_micro']  # the metrics we use

        metrics = {}
        for score in scores:
            metrics.update({
                score: cross_val_score(_model, self.X_train, self.y_train, cv=k_fold, n_jobs=-1, scoring=score).mean()
            })

        mean_tpr = 0.0  # true positive rate
        mean_fpr = np.linspace(0, 1, 100)  # false positive rate
        for train_index, test_index in k_fold.split(self.X_train, self.y_train):
            _X_train, _X_test = self.X_train[train_index], self.X_train[test_index]
            _y_train, _y_test = self.y_train[train_index], self.y_train[test_index]

            clf = OneVsRestClassifier(_model, -1)
            clf.fit(_X_train, _y_train)
            if hasattr(clf, "predict_proba"):
                probas = clf.predict_proba(_X_test)
                probas = probas[:, 1]
            else:
                probas = clf.decision_function(_X_test)
                probas = (probas - probas.min()) / (probas.max() - probas.min())

            fpr, tpr, _ = roc_curve(_y_test.ravel(), probas.ravel())
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

        mean_tpr /= k_fold.get_n_splits()
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        metrics.update({'roc_tpr_micro': mean_tpr, 'roc_fpr_micro': mean_fpr, 'roc_auc_micro': mean_auc})

        t2 = time()

        print(self.model_name + " = " + str(
            round(t2 - t1)) + "s")
        print("________________________________________")

        self.model = _model
        self.metrics = metrics
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
            self.load_trained_model(full_model_name)
        else:
            self.train_model(self.create_pipeline())

    def load_trained_model(self, name):
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
    def create_features(data, times=10):
        data['n_times_title'] = ''
        for i in range(0, times):
            data['n_times_title'] = data['n_times_title'] + ' ' + data['title']
        data['total'] = data['n_times_title'] + ' ' + data['text']
        data.drop('n_times_title', axis=1, inplace=True)
        return data
