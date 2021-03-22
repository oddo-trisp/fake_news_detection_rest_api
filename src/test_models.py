import traceback

from src.models.FakeNewsClassifier import FakeNewsClassifier
from src.models.SupervisedLearner import SupervisedLearner
from src.utils.conf import *


def generic_test(clf_names=None, clf_feature_names=None, language=ENGLISH):
    try:
        metrics_scores = {}

        if clf_names is None:
            clf_names = CLASSIFICATION_SET

        if clf_feature_names is None:
            clf_feature_names = CLASSIFICATION_FEATURE_SET

        for clf_name in clf_names:
            for clf_feature_name in clf_feature_names:
                clf = FakeNewsClassifier(clf_name, clf_feature_name, True, language)
                metrics_scores.update({clf.model_name: clf.metrics})

            SupervisedLearner.plot_roc_curve(metrics_scores, clf_name)
            SupervisedLearner.plot_mae_curve(metrics_scores, clf_name)
            SupervisedLearner.save_metrics_to_csv(metrics_scores, clf_name)
            metrics_scores.clear()

    except Exception:
        error_string = traceback.format_exc()
        print(error_string)


def total_test():
    language = GREEK
    metrics_scores = {}

    clf = FakeNewsClassifier(ADA_BOOST, TF_IDF, True, language)
    metrics_scores.update({clf.model_name: clf.metrics})

    clf = FakeNewsClassifier(LOGISTIC_REGRESSION, TF_IDF, True, language)
    metrics_scores.update({clf.model_name: clf.metrics})

    clf = FakeNewsClassifier(EXTRA_TREES, TF_IDF, True, language)
    metrics_scores.update({clf.model_name: clf.metrics})

    clf = FakeNewsClassifier(RANDOM_FOREST, TF_IDF, True, language)
    metrics_scores.update({clf.model_name: clf.metrics})

    clf = FakeNewsClassifier(SVM, TF_IDF, True, language)
    metrics_scores.update({clf.model_name: clf.metrics})

    clf = FakeNewsClassifier(KNN, TRUNC_SVD, True, language)
    metrics_scores.update({clf.model_name: clf.metrics})

    SupervisedLearner.plot_roc_curve(metrics_scores, 'total')
    SupervisedLearner.save_metrics_to_csv(metrics_scores, 'total')


generic_test(clf_names=[LOGISTIC_REGRESSION, ADA_BOOST, EXTRA_TREES, RANDOM_FOREST, SVM, KNN],
             clf_feature_names=[BOW, TF_IDF, TRUNC_SVD, W2V],
             language=GREEK)
