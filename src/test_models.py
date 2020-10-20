import smtplib
import traceback
from email.message import EmailMessage

from src.models.FakeNewsClassifier import FakeNewsClassifier
from src.models.FakeNewsDeepLearner import FakeNewsDeepLearner
from src.models.SupervisedLearner import SupervisedLearner
from src.utils.conf import *


def generic_test(clf_names=None, nn_names=None, clf_feature_names=None, nn_feature_names=None, language=ENGLISH):
    server = crete_connection_to_server()

    try:
        metrics_scores = {}

        if clf_names is None:
            clf_names = CLASSIFICATION_SET

        if nn_names is None:
            nn_names = DEEP_LEARNING_SET

        if clf_feature_names is None:
            clf_feature_names = CLASSIFICATION_FEATURE_SET

        if nn_feature_names is None:
            nn_feature_names = DEEP_LEARNING_FEATURE_SET

        for clf_name in clf_names:
            for clf_feature_name in clf_feature_names:
                # server = get_active_server(server)
                # send_email(server, clf_name + '_' + clf_feature_name + ' started!', 'Training Start')

                clf = FakeNewsClassifier(clf_name, clf_feature_name, True, language)
                metrics_scores.update({clf.model_name: clf.metrics})

                # server = get_active_server(server)
                # send_email(server, clf_name + '_' + clf_feature_name + ' finished!', 'Training Success')
            SupervisedLearner.plot_roc_curve(metrics_scores, clf_name)
            SupervisedLearner.save_metrics_to_csv(metrics_scores, clf_name)
            metrics_scores.clear()

        for nn_name in nn_names:
            for nn_feature_name in nn_feature_names:
                # server = get_active_server(server)
                # send_email(server, nn_name + '_' + nn_feature_name + ' started!', 'Training Start')

                nn = FakeNewsDeepLearner(nn_name, nn_feature_name, True, language)
                metrics_scores.update({nn.model_name: nn.metrics})

                # server = get_active_server(server)
                # send_email(server, nn_name + '_' + nn_feature_name + ' finished!', 'Training Success')
            SupervisedLearner.plot_roc_curve(metrics_scores, nn_name)
            SupervisedLearner.save_metrics_to_csv(metrics_scores, nn_name)
            metrics_scores.clear()

    except Exception as e:
        error_string = traceback.format_exc()
        server = get_active_server(server)
        send_email(server, str(e) + '\n' + error_string, 'Training Error')
        print(error_string)
    finally:
        server.quit()


def crete_connection_to_server():
    server = smtplib.SMTP('smtp.uoa.gr', 587)
    server.starttls()
    server.login('odytrisp', 'John_Lennon1980')
    return server


def is_connected(server):
    try:
        status = server.noop()[0]
    except:  # smtplib.SMTPServerDisconnected
        status = -1
    return True if status == 250 else False


def get_active_server(server):
    return server if is_connected(server) else crete_connection_to_server()


def send_email(server, message, subject):
    email = EmailMessage()
    email.set_content(message, subtype='html')
    email['From'] = "odytrisp@di.uoa.gr"
    email['To'] = "odytrisp@di.uoa.gr"
    email['Subject'] = subject
    server.send_message(email)


def resave_corrupted_files():
    for clf_name in [ADA_BOOST, EXTRA_TREES, RANDOM_FOREST, KNN]:
        clf = FakeNewsClassifier(clf_name, TRUNC_SVD, True, GREEK)
        clf.save_model()


generic_test(clf_names=[ADA_BOOST, EXTRA_TREES, RANDOM_FOREST, KNN, LOGISTIC_REGRESSION],
             nn_names=[CNN, RNN],
             clf_feature_names=[BOW, TF_IDF, TRUNC_SVD, W2V],
             nn_feature_names=[PAD_SEQ, W2V],
             language=GREEK)

# resave_corrupted_files()
