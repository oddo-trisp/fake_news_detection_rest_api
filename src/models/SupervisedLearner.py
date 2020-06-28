import datetime
import re
from abc import abstractmethod
from itertools import cycle
from os import path
from time import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

import src.utils.utils as utils
from src.models.ISupervisedLearner import ISupervisedLearner
from src.utils.conf import *


class SupervisedLearner(ISupervisedLearner):

    def __init__(self, learner_name, feature_name, evaluate, language=ENGLISH, df_train=None, df_test=None):

        self.test_id = None
        self.X_train = None
        self.y_train = None
        self.X_test = None

        self.metrics = None
        self.model = None

        self.language = language
        self.evaluate = evaluate

        self.learner_name = None
        self.feature_name = None
        self.model_name = None
        self.model_path = None
        self.params_path = None
        self.metrics_path = None

        self.init_paths(learner_name, feature_name)
        self.prepare_data(df_train, df_test)

        self.init_model()

    def validate_init(self, learner_name, feature_name):
        if learner_name is None or feature_name is None:
            raise Exception('learner_name and feature_name should not be None')

    def init_paths(self, learner_name, feature_name):
        self.learner_name = learner_name
        self.feature_name = feature_name
        self.model_name = learner_name + '_' + feature_name
        self.model_path = utils.get_valid_path(MODELS_PATH + self.model_name + FORMAT_SAV)
        self.params_path = utils.get_valid_path(PARAMS_PATH + self.model_name + FORMAT_PICKLE)
        self.metrics_path = utils.get_valid_path(METRICS_PATH + self.model_name + FORMAT_PICKLE)

    def prepare_data(self, df_train=None, df_test=None):
        if df_train is None:
            df_train = utils.read_csv(utils.get_valid_path(TRAIN_PATH))
        if df_test is None:
            df_test = utils.read_csv(utils.get_valid_path(TEST_PATH))

        self.prepare_train_data(df_train)
        self.prepare_test_data(df_test)

    def prepare_train_data(self, df_train):
        df_train = self.engineer_data(df_train)

        self.X_train = df_train['total'].values
        self.y_train = df_train['label'].values

    def prepare_test_data(self, df_test):
        self.test_id = df_test['id']
        df_test['label'] = 't'
        df_test = self.engineer_data(df_test, remove_outliers=False)

        self.X_test = df_test['total'].values

    def train_model(self):
        t1 = time()

        model, metrics = self.evaluate_model() if self.evaluate \
            else self.simple_train_model()

        t2 = time()

        print(self.model_name + " = " + str(
            round(t2 - t1)) + "s")
        print("________________________________________")

        self.model = model
        self.metrics = metrics

        self.save_model()
        self.save_metrics()

    def evaluate_model(self):
        metrics = {}
        model = self.create_pipeline()

        model = self.hyperparameters_evaluation(model)
        model, metrics = self.k_fold_evaluation(model, metrics)

        return model, metrics

    def hyperparameters_evaluation(self, model):
        best_parameters = self.get_model_params()

        if best_parameters is None:
            # Evaluate model using grid search
            print("\nPerforming grid search for  " + self.model_name + " learner")

            k_fold, scores = self.get_k_fold()
            fit_params = self.get_fit_params()
            parameters = self.get_evaluation_params()
            n_jobs = self.get_n_jobs()

            t0 = time()

            grid_search = GridSearchCV(model, parameters, cv=k_fold.get_n_splits(), scoring=scores, n_jobs=n_jobs,
                                       refit='accuracy', verbose=VERBOSE)
            grid_search.fit(self.X_train, self.y_train, **fit_params)

            t1 = time()

            print(grid_search.cv_results_)
            print("Best model tuning\n")
            print(grid_search.best_score_)
            print(grid_search.best_params_)
            print("Done in " + str(round(t1 - t0)) + "s")

            best_parameters = grid_search.best_params_
            utils.save_pickle_file(self.params_path, best_parameters)

        model.set_params(**best_parameters)

        return model

    def k_fold_evaluation(self, model, metrics):
        print("\nRunning 10-Fold test for: " + str(self.model_name))  # running prompt explaining which algorithm runs

        k_fold, scores = self.get_k_fold()
        fit_params = self.get_fit_params()
        n_jobs = self.get_n_jobs()

        t0 = time()

        for score in scores:
            metrics.update({
                score: cross_val_score(model, self.X_train, self.y_train, cv=k_fold, n_jobs=n_jobs,
                                       scoring=score, verbose=VERBOSE, fit_params=fit_params).mean()
            })

        mean_tpr = np.linspace(0, 0, 100)  # true positive rate
        mean_fpr = np.linspace(0, 1, 100)  # false positive rate
        for train_index, test_index in k_fold.split(self.X_train, self.y_train):
            _X_train, _X_test = self.X_train[train_index], self.X_train[test_index]
            _y_train, _y_test = self.y_train[train_index], self.y_train[test_index]

            model.fit(_X_train, _y_train, **fit_params)
            if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
                probas = model.predict_proba(_X_test)
                probas = probas[:, 1]
            else:
                probas = model.decision_function(_X_test)
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

        return model, metrics

    def simple_train_model(self):
        model = self.create_pipeline()
        params = self.get_model_params()
        model.set_params(**params)

        metrics = {}

        k_fold, scores = self.get_k_fold()  # a KFold variation

        for train_index, test_index in k_fold.split(self.X_train, self.y_train):
            _X_train, _X_test = self.X_train[train_index], self.X_train[test_index]
            _y_train, _y_test = self.y_train[train_index], self.y_train[test_index]

            model.fit(_X_train, _y_train)

        return model, metrics

    def predict_proba(self, df_test=None):
        if df_test is not None:
            self.prepare_test_data(df_test)

        start_time = time()

        y_test = self.model.predict_proba(self.X_test)
        y_test = np.asanyarray([['%.5f' % elem for elem in subarray] for subarray in y_test])
        y_test = utils.create_dataframe(y_test, columns=['0', '1'])

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
        self.model = self.load_model()
        self.metrics = self.load_metrics()

    def create_pipeline(self, learner=None, features=None):
        learner = self.create_learner() if learner is None else learner
        features = self.create_features() if features is None else features

        steps = []
        for k, v in features.items():
            if v is not None:
                steps.append((k, v))

        for k, v in learner.items():
            if v is not None:
                steps.append((k, v))

        pipeline = Pipeline(steps=steps)

        return pipeline

    @abstractmethod
    def create_learner(self):
        pass

    @abstractmethod
    def create_features(self):
        pass

    @abstractmethod
    def get_evaluation_params(self):
        pass

    def get_model_params(self):
        params = None
        if path.exists(self.params_path):
            params = utils.load_pickle_file(self.params_path)

        return params

    def save_model(self):
        utils.save_pickle_file(self.model_path, self.model)

    def load_model(self):
        return utils.load_pickle_file(self.model_path)

    def save_metrics(self):
        utils.save_pickle_file(self.metrics_path, self.metrics)

    def load_metrics(self):
        return utils.load_pickle_file(self.metrics_path)

    def get_stopwords(self):
        try:
            stop_words = stopwords.words(self.language)
        except LookupError:
            nltk.download('stopwords')
            stop_words = stopwords.words(self.language)

        return set(stop_words)

    def get_n_jobs(self):
        return 1 if self.feature_name in DEEP_LEARNING_FEATURE_SET or self.learner_name in {EXTRA_TREES} else -1

    @staticmethod
    @abstractmethod
    def get_fit_params():
        pass

    @staticmethod
    def add_clf_prefix(params):
        return {f'{CLF}__{k}': v for k, v in params.items()}

    @staticmethod
    def get_k_fold():
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=434)  # a KFold variation
        scores = ['accuracy', 'precision_micro', 'recall_micro']  # the metrics we use
        return k_fold, scores

    @staticmethod
    def save_prediction_to_csv(filename, test_id, y_test):
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        utils.save_csv_file(file_path=utils.get_valid_path(OUTPUT_PATH + filename + "_" + now + FORMAT_CSV),
                            data=y_test, columns=['0', '1'], id_column=test_id, sep=',')

    @staticmethod
    def engineer_data(data, remove_outliers=True):
        data = SupervisedLearner.remove_useless_columns(data)

        if remove_outliers is True:
            data = SupervisedLearner.fill_na_values(data, columns=['title'])
            data = SupervisedLearner.remove_na_values(data)
            data = SupervisedLearner.remove_outliers(data)
        else:
            data = SupervisedLearner.fill_na_values(data)

        data = SupervisedLearner.create_new_columns(data)
        data = SupervisedLearner.remove_noise(data)

        return data

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
    def remove_noise(data):
        text = data['total']
        rules = [
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
            {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
            {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
            {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
            {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
            {r'^\s+': u''}  # remove spaces at the beginning
        ]
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text.str.replace(regex, v)
        text = text.str.rstrip()
        data['total'] = text
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
        plt.legend(loc='lower right')  # legend position
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
        utils.save_csv_file(file_path=utils.get_valid_path(EVALUATION_METRIC_PATH), data=metrics,
                            index=proper_labels, columns=metrics.keys(), float_format='%.3f', sep='\t')

    @staticmethod
    def rename_labels(metrics, label, proper_label):
        metrics[proper_label] = metrics[label]
        del metrics[label]
