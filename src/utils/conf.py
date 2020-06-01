# Resource related properties

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

# Languages
ENGLISH = 'english'
GREEK = 'greek'

# Classifiers
EXTRA_TREES = 'ExtraTrees'
ADA_BOOST = 'AdaBoost'
GAUSSIAN_NB = 'GaussianNB'
LOGISTIC_REGRESSION = 'LogisticRegression'
RANDOM_FOREST = 'RandomForest'
SVM = 'SupportVectorMachine'

W2V_MODEL = 'Word2Vec'

# Features
BOW = 'BoW'
SVD = 'SVD'
W2V = 'W2V'

# Neural Networks
LSTM = 'LSTM'
GRU = 'GRU'

# TODO: Check if values should be the same for deep learning
N_ESTIMATORS = 20
MAX_FEATURES = 100
N_GRAM_RANGE = (1, 2)

CLASSIFICATION_SET = {EXTRA_TREES, ADA_BOOST, GAUSSIAN_NB, LOGISTIC_REGRESSION, RANDOM_FOREST, SVM}
DEEP_LEARNING_SET = {LSTM, GRU}
FEATURE_SET = {BOW, SVD, W2V}
