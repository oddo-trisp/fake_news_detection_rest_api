import datetime
from abc import abstractmethod
from itertools import cycle
from os import path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

import nltk
from nltk.corpus import stopwords

import src.utils.utils as utils
from src.models.ISupervisedLearner import ISupervisedLearner
from src.utils.conf import *


class SupervisedLearner(ISupervisedLearner):

    def __init__(self, _learner_name, _feature_name, _evaluate=False, _df_test=None, _df_train=None):

        super().__init__(_learner_name, _feature_name, _evaluate, _df_test, _df_train)
        if _learner_name is None or _feature_name is None:
            return

        self.test_id = None

        self.X_train = None
        self.y_train = None

        # self.X_test_clear = None
        self.X_test = None

        self.metrics = None
        self.model = None

        self.evaluate = _evaluate
        self.learner_name = _learner_name
        self.feature_name = _feature_name
        self.model_name = _learner_name + '_' + _feature_name
        self.model_path = utils.get_valid_path(PIPELINE_PATH + self.model_name + FORMAT_SAV)
        self.metrics_path = utils.get_valid_path(METRICS_PATH + self.model_name + FORMAT_SAV)

        self.prepare_data(_df_train, _df_test)

        self.init_model()

    @abstractmethod
    def prepare_data(self, df_train=None, df_test=None):
        pass

    @abstractmethod
    def prepare_train_data(self, df_train):
        pass

    @abstractmethod
    def prepare_test_data(self, df_test):
        pass

    def train_model(self):
        t1 = time()

        _model, metrics = self.evaluate_model() if self.evaluate \
            else self.simple_train_model()

        t2 = time()

        print(self.model_name + " = " + str(
            round(t2 - t1)) + "s")
        print("________________________________________")

        self.model = _model
        self.metrics = metrics

        utils.save_file(self.model_path, self.model)
        utils.save_file(self.metrics_path, self.metrics)

    def evaluate_model(self):
        metrics = {}

        # TODO remove comment for full evaluation
        # _model = self.hyperparameters_evaluation()
        _model = self.create_pipeline()
        _model, metrics = self.k_fold_evaluation(_model, metrics)

        return _model, metrics

    def hyperparameters_evaluation(self):
        # Evaluate classifier using grid search
        print("\nPerforming grid search for  " + self.learner_name + " learner")

        learner, parameters = self.create_default_learner()
        k_fold, scores = self.get_k_fold()

        t0 = time()

        grid_search = GridSearchCV(learner, parameters, cv=10, scoring=scores, n_jobs=-1, refit='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        t1 = time()

        print(grid_search.cv_results_)
        print(learner + " best model tuning\n")
        print(grid_search.best_score_)
        print(grid_search.best_params_)
        print(grid_search.best_estimator_)
        print("Done in " + str(round(t1 - t0)) + "s")

        learner = grid_search.best_estimator_
        _model = self.create_pipeline(learner=learner)

        return _model

    def k_fold_evaluation(self, _model, metrics):
        print("\nRunning 10-Fold test for: " + str(self.model_name))  # running prompt explaining which algorithm runs

        k_fold, scores = self.get_k_fold()
        n_jobs = 1 if self.feature_name is W2V or self.learner_name is EXTRA_TREES else -1

        t0 = time()

        for score in scores:
            metrics.update({
                score: cross_val_score(_model, self.X_train, self.y_train, cv=k_fold, n_jobs=n_jobs,
                                       scoring=score).mean()
            })

        mean_tpr = np.linspace(0, 0, 100)  # true positive rate
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

        t1 = time()

        print("Done in " + str(round(t1 - t0)) + "s")

        return _model, metrics

    def simple_train_model(self):
        _model = self.create_pipeline(None, None)
        metrics = {}

        k_fold, scores = self.get_k_fold()  # a KFold variation

        for train_index, test_index in k_fold.split(self.X_train, self.y_train):
            _X_train, _X_test = self.X_train[train_index], self.X_train[test_index]
            _y_train, _y_test = self.y_train[train_index], self.y_train[test_index]

            _model.fit(_X_train, _y_train)

        return _model, metrics

    def predict_proba(self, _df_test=None):
        if _df_test is not None:
            self.prepare_test_data(_df_test)

        start_time = time()

        y_test = self.model.predict_proba(self.X_test)
        y_test = np.asanyarray([['%.5f' % elem for elem in subarray] for subarray in y_test])
        y_test = pd.DataFrame(y_test, columns=['0', '1'])

        self.save_prediction_to_csv(self.model_name, self.test_id, y_test)
        print("Time to Predict ({}): {:.4f}s \n".format(self.model_name, time() - start_time))

        result = y_test['1'].values[0]
        return result

    def init_model(self):
        if path.exists(self.model_path) and path.exists(self.metrics_path):
            self.load_trained_model()
        else:
            self.train_model()

    def load_trained_model(self):
        self.model = utils.load_file(self.model_path)
        self.metrics = utils.load_file(self.metrics_path)

    @abstractmethod
    def create_pipeline(self, learner=None, features=None):
        pass

    @abstractmethod
    def create_learner(self):
        pass

    @abstractmethod
    def create_default_learner(self):
        pass

    @abstractmethod
    def create_features(self):
        pass

    @staticmethod
    def get_k_fold():
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=434)  # a KFold variation
        scores = ['accuracy', 'precision_micro', 'recall_micro']  # the metrics we use
        return k_fold, scores

    @staticmethod
    def get_stopwords():
        try:
            stop_words = stopwords.words("english")
        except LookupError:
            nltk.download('stopwords')
            stop_words = stopwords.words("english")

        return set(stop_words)

    @staticmethod
    def save_prediction_to_csv(filename, test_id, y_test):
        sub = pd.DataFrame(y_test, columns=['0', '1'])
        sub['id'] = test_id
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        sub.to_csv(utils.get_valid_path(OUTPUT_PATH + filename + "_" + now + FORMAT_CSV), index=False, sep=',')

    @staticmethod
    @abstractmethod
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
        plt.savefig(utils.get_valid_path(ROC_PLOT_PATH), bbox='tight')  # save the png file

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
        df.to_csv(utils.get_valid_path(EVALUATION_METRIC_PATH), sep='\t', float_format="%.3f")

    @staticmethod
    def rename_labels(metrics, label, proper_label):
        metrics[proper_label] = metrics[label]
        del metrics[label]
