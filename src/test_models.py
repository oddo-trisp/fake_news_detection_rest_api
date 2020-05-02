from src.models.FakeNewsClassifier import FakeNewsClassifier
from src.utils.conf import *


def generic_test(clf_names=None, feature_names=None, language=ENGLISH):
    metrics_scores = {}

    if clf_names is None:
        clf_names = CLASSIFICATION_SET
    if feature_names is None:
        feature_names = FEATURE_SET

    for clf_name in clf_names:
        for feature_name in feature_names:
            clf = FakeNewsClassifier(clf_name, feature_name, True, language)
            metrics_scores.update({clf.model_name: clf.metrics})

    FakeNewsClassifier.plot_roc_curve(metrics_scores)
    FakeNewsClassifier.save_metrics_to_csv(metrics_scores)


# generic_test(clf_names={RANDOM_FOREST, LOGISTIC_REGRESSION,
#                         ADA_BOOST, EXTRA_TREES, GAUSSIAN_NB})

generic_test(clf_names={RANDOM_FOREST}, feature_names={W2V}, language=ENGLISH)
