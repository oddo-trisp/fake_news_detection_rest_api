from src.models.FakeNewsClassifier import FakeNewsClassifier
from src.models.FakeNewsDeepLearner import FakeNewsDeepLearner
from src.models.SupervisedLearner import SupervisedLearner
from src.utils.conf import *


def generic_test(clf_names=None, nn_names=None, feature_names=None, language=ENGLISH):
    metrics_scores = {}

    if clf_names is None:
        clf_names = CLASSIFICATION_SET

    if nn_names is None:
        nn_names = DEEP_LEARNING_SET

    if feature_names is None:
        feature_names = FEATURE_SET

    for feature_name in feature_names:
        # for clf_name in clf_names:
        #     clf = FakeNewsClassifier(clf_name, feature_name, True, language)
        #     metrics_scores.update({clf.model_name: clf.metrics})
        for nn_name in nn_names:
            nn = FakeNewsDeepLearner(nn_name, feature_name, True, language)
            metrics_scores.update({nn.model_name: nn.metrics})

    SupervisedLearner.plot_roc_curve(metrics_scores)
    SupervisedLearner.save_metrics_to_csv(metrics_scores)


# generic_test(clf_names={RANDOM_FOREST, LOGISTIC_REGRESSION,
#                         ADA_BOOST, EXTRA_TREES, GAUSSIAN_NB})

generic_test(clf_names={RANDOM_FOREST}, nn_names={LSTM_NN}, feature_names={W2V}, language=ENGLISH)
