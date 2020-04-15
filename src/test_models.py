from src.models.FakeNewsClassifier import FakeNewsClassifier
from src.utils.conf import *


def test_models():
    metrics_scores = {}

    # Random Forest Test
    random_forest_test(metrics_scores)

    # Logistic Regression Test
    logistic_regression_test(metrics_scores)

    # AdaBoost Test
    adaboost_test(metrics_scores)

    FakeNewsClassifier.plot_roc_curve(metrics_scores)
    FakeNewsClassifier.save_metrics_to_csv(metrics_scores)


def random_forest_test(metrics_scores):
    random_forest_bow = FakeNewsClassifier(RANDOM_FOREST, BOW)
    metrics_scores.update({random_forest_bow.model_name: random_forest_bow.metrics})

    random_forest_svd = FakeNewsClassifier(RANDOM_FOREST, SVD)
    metrics_scores.update({random_forest_svd.model_name: random_forest_svd.metrics})

    random_forest_w2v = FakeNewsClassifier(RANDOM_FOREST, W2V)
    metrics_scores.update({random_forest_w2v.model_name: random_forest_w2v.metrics})


def logistic_regression_test(metrics_scores):
    log_reg_bow = FakeNewsClassifier(LOGISTIC_REGRESSION, BOW)
    metrics_scores.update({log_reg_bow.model_name: log_reg_bow.metrics})

    log_reg_svd = FakeNewsClassifier(LOGISTIC_REGRESSION, SVD)
    metrics_scores.update({log_reg_svd.model_name: log_reg_svd.metrics})

    log_reg_w2v = FakeNewsClassifier(LOGISTIC_REGRESSION, W2V)
    metrics_scores.update({log_reg_w2v.model_name: log_reg_w2v.metrics})


def adaboost_test(metrics_scores):
    ada_boost_bow = FakeNewsClassifier(ADA_BOOST, BOW)
    metrics_scores.update({ada_boost_bow.model_name: ada_boost_bow.metrics})

    ada_boost_svd = FakeNewsClassifier(ADA_BOOST, SVD)
    metrics_scores.update({ada_boost_svd.model_name: ada_boost_svd.metrics})

    ada_boost_w2v = FakeNewsClassifier(ADA_BOOST, W2V)
    metrics_scores.update({ada_boost_w2v.model_name: ada_boost_w2v.metrics})


test_models()
