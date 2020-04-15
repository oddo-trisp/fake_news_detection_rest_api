import datetime
import pickle
from itertools import cycle
from os import path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.ISupervisedLearner import ISupervisedLearner
from src.utils.conf import *


class SupervisedLearner(ISupervisedLearner):

    def __init__(self, _learner_name=None, _feature_name=None, _df_train=None, _df_test=None):

        super().__init__(_learner_name, _feature_name, _df_train, _df_test)
        if _learner_name is None or _feature_name is None:
            return

        self.test_id = None

        self.X_train = None
        self.y_train = None

        # self.X_test_clear = None
        self.X_test = None

        self.metrics = None
        self.model = None

        self.learner_name = _learner_name
        self.feature_name = _feature_name
        self.model_name = _learner_name + '_' + _feature_name
        self.model_path = get_valid_path(PIPELINE_PATH + self.model_name + FORMAT_SAV)
        self.metrics_path = get_valid_path(METRICS_PATH + self.model_name + FORMAT_SAV)

        self.prepare_data(_df_train, _df_test)

        self.init_model()

    def prepare_data(self, df_train=None, df_test=None):
        pass

    def prepare_train_data(self, df_train):
        pass

    def prepare_test_data(self, df_test):
        pass

    def train_model(self, _model):
        t1 = time()

        print("\nRunning 10-Fold test for: " + str(self.model_name))  # running prompt explaining which algorithm runs

        k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=434)  # a KFold variation
        scores = ['accuracy', 'precision_micro', 'recall_micro']  # the metrics we use
        n_jobs = 1 if self.feature_name is W2V else -1

        metrics = {}
        for score in scores:
            metrics.update({
                score: cross_val_score(_model, self.X_train, self.y_train, cv=k_fold, n_jobs=n_jobs,
                                       scoring=score).mean()
            })

        mean_tpr = 0.0  # true positive rate
        mean_fpr = np.linspace(0, 1, 100)  # false positive rate
        for train_index, test_index in k_fold.split(self.X_train, self.y_train):
            _X_train, _X_test = self.X_train[train_index], self.X_train[test_index]
            _y_train, _y_test = self.y_train[train_index], self.y_train[test_index]

            _model.fit(_X_train, _y_train)
            if hasattr(_model, "predict_proba"):
                probas = _model.predict_proba(_X_test)
                probas = probas[:, 1]
            else:
                probas = _model.decision_function(_X_test)
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

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(self.metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)

    def predict_proba(self, _df_test=None):
        if _df_test is not None:
            self.prepare_test_data(_df_test)

        start_time = time()

        y_test = self.model.predict_proba(self.X_test)
        y_test = np.asanyarray([['%.5f' % elem for elem in subarray] for subarray in y_test])
        y_test = pd.DataFrame(y_test, columns=['0', '1'])

        SupervisedLearner.save_prediction_to_csv(self.model_name, self.test_id, y_test)
        print("Time to Predict ({}): {:.4f}s \n".format(self.model_name, time() - start_time))

        result = y_test['1'].values[0]
        return result

    def init_model(self):
        if path.exists(self.model_path) and path.exists(self.metrics_path):
            self.load_trained_model()
        else:
            self.train_model(self.create_pipeline())

    def load_trained_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(self.metrics_path, 'rb') as f:
            self.metrics = pickle.load(f)

    def create_pipeline(self):
        pass

    def create_learner(self):
        pass

    def create_features(self):
        pass

    @staticmethod
    def save_prediction_to_csv(filename, test_id, y_test):
        sub = pd.DataFrame(y_test, columns=['0', '1'])
        sub['id'] = test_id
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        sub.to_csv(get_valid_path(OUTPUT_PATH + filename + "_" + now + FORMAT_CSV), index=False, sep=',')

    @staticmethod
    def engineer_data(data, remove_outliers=True):
        pass

    @staticmethod
    def remove_outliers(data):
        data = data.drop(data[data['text'].map(len) < 50].index, axis=0)
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
    def create_new_columns(data, times=10):
        data['n_times_title'] = ''
        for i in range(0, times):
            data['n_times_title'] = data['n_times_title'] + ' ' + data['title']
        data['total'] = data['n_times_title'] + ' ' + data['text']
        data = data.drop('n_times_title', axis=1)
        return data

    @staticmethod
    def plot_roc_curve(metrics):
        colors = cycle(['aqua', 'indigo', 'seagreen', 'crimson', 'teal', 'olive'])
        lw = 1.25  # line width
        plt.figure()
        for class_name, color in zip(metrics, colors):  # different color for every metric
            metric = metrics[class_name]
            plt.plot(metric['roc_fpr_micro'], metric['roc_tpr_micro'], color=color, lw=lw,
                     label='{0} (AUC = {1:0.2f})'
                           ''.format(class_name, metric['roc_auc_micro']))  # plot fpr, tpr and print auc
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
        plt.xlim([-0.05, 1.05])  # x limit
        plt.ylim([-0.05, 1.05])  # y limit
        plt.xlabel('False Positive Rate')  # x label
        plt.ylabel('True Positive Rate')  # y label
        plt.title('ROC curves of Different Classifiers')  # plot title
        plt.legend(loc="lower right")  # legend position
        plt.savefig(get_valid_path(ROC_PLOT_PATH), bbox='tight')  # save the png file

    @staticmethod
    def save_metrics_to_csv(metrics):
        # Create pandas DataFrame to generate EvaluationMetric 10-fold CSV output.
        proper_labels = ['Accuracy', 'Precision', 'Recall']

        for k, v in metrics.items():
            # Delete useless columns
            del metrics[k]['roc_fpr_micro']
            del metrics[k]['roc_tpr_micro']
            del metrics[k]['roc_auc_micro']

            # Rename metrics to proper ones
            SupervisedLearner.rename_labels(metrics[k], 'accuracy', proper_labels[0])

            # Rename metrics to proper ones
            SupervisedLearner.rename_labels(metrics[k], 'precision_micro', proper_labels[1])

            # Rename metrics to proper ones
            SupervisedLearner.rename_labels(metrics[k], 'recall_micro', proper_labels[2])

        # generate CSV output
        df = pd.DataFrame(metrics, index=proper_labels, columns=metrics.keys())
        df.to_csv(get_valid_path(EVALUATION_METRIC_PATH), sep='\t', float_format="%.3f")

    @staticmethod
    def rename_labels(metrics, label, proper_label):
        metrics[proper_label] = metrics[label]
        del metrics[label]
