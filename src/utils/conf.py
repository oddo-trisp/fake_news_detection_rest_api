# Resource related properties
import os
from pathlib import Path

TRAIN_DATASET = 'train.csv'
TEST_DATASET = 'test.csv'
ROC_PLOT_FILE = 'roc_10fold.png'
EVALUATION_METRIC_FILE = 'evaluation_metrics_10fold.csv'

RESOURCES_PATH = 'resources/'

INPUT_PATH = RESOURCES_PATH + 'input/'
OUTPUT_PATH = RESOURCES_PATH + 'output/'
PIPELINE_PATH = RESOURCES_PATH + 'pipeline/'
METRICS_PATH = RESOURCES_PATH + 'metrics/'

TRAIN_PATH = INPUT_PATH + TRAIN_DATASET
TEST_PATH = INPUT_PATH + TEST_DATASET
ROC_PLOT_PATH = OUTPUT_PATH + ROC_PLOT_FILE
EVALUATION_METRIC_PATH = OUTPUT_PATH + EVALUATION_METRIC_FILE

FORMAT_CSV = '.csv'
FORMAT_SAV = '.sav'

# Classifiers
EXTRA_TREES = 'ExtraTrees'
ADA_BOOST = 'AdaBoost'
MULTINOMIAL_NB = 'MultinomialNB'
LOGISTIC_REGRESSION = 'LogisticRegression'
RANDOM_FOREST = 'RandomForest'
SVM = 'SupportVectorMachine'

# Features
BOW = 'BoW'
SVD = 'SVD'
W2V = 'W2V'

# Neural Networks
LSTM = "LSTM"
GRU = "GRU"

N_ESTIMATORS = 20
MAX_FEATURES = 100
N_GRAM_RANGE = (1, 2)

CLASSIFICATION_SET = {EXTRA_TREES, ADA_BOOST, RANDOM_FOREST, MULTINOMIAL_NB, LOGISTIC_REGRESSION}
DEEP_LEARNING_SET = {LSTM, GRU}


def get_valid_path(destination):
    root_path = Path(__file__).parent.parent.parent
    working_dir = os.getcwd()
    steps = Path(working_dir).relative_to(root_path).as_posix().count('/') + 1

    prefix = ''
    for i in range(steps):
        prefix = prefix + '../'

    return Path(prefix + destination)
