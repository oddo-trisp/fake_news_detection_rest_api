from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

import pydotplus
from tensorflow.python.keras.utils.vis_utils import model_to_dot
import src.utils.utils as utils
from src.models.SupervisedLearner import SupervisedLearner
from src.models.transformers import AverageWordVectorTransformer, OneHotTransformer
from src.utils.conf import *


class FakeNewsDeepLearner(SupervisedLearner):

    def __init__(self, learner_name, feature_name, evaluate, language=ENGLISH, df_train=None, df_test=None):
        self.validate_init(learner_name, feature_name)

        self.keras_model_path = None
        self.vocabulary_size = None
        self.embedding_size = None
        self.max_length = None

        super().__init__(learner_name, feature_name, evaluate, language, df_train, df_test)

    def validate_init(self, learner_name, feature_name):
        super().validate_init(learner_name, feature_name)
        if learner_name not in DEEP_LEARNING_SET or feature_name not in DEEP_LEARNING_FEATURE_SET:
            raise Exception('learner_name and feature_name must belong to deep learning set')

    def init_paths(self, learner_name, feature_name):
        super().init_paths(learner_name, feature_name)
        keras_model_name = KERAS + '_' + self.model_name
        self.keras_model_path = utils.get_valid_path(MODELS_PATH + keras_model_name + FORMAT_H5)

    def create_learner(self):
        nn_clf = KerasClassifier(build_fn=self.create_nn_model, verbose=VERBOSE)
        return {CLF: nn_clf}

    def create_nn_model(self, optimizer='adam'):
        nn_model = None

        if self.learner_name is RNN:
            # LSTM Neural Network
            nn_model = Sequential()
            nn_model.add(Embedding(self.vocabulary_size, self.embedding_size, input_length=self.max_length))
            nn_model.add(LSTM(100))
            nn_model.add(Dense(1, activation='sigmoid'))
        elif self.learner_name is CNN:
            # Convolution neural Network
            nn_model = Sequential()
            nn_model.add(Embedding(self.vocabulary_size, self.embedding_size, input_length=self.max_length))
            nn_model.add(Conv1D(128, 5, activation='relu'))
            nn_model.add(GlobalMaxPooling1D())
            nn_model.add(Dense(10, activation='relu'))
            nn_model.add(Dense(1, activation='sigmoid'))

        nn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print(nn_model.summary())
        return nn_model

    def create_features(self):
        vect = None

        stop_words = self.get_stopwords()

        if self.feature_name is W2V:
            vect = AverageWordVectorTransformer(language=self.language, stop_words=stop_words,
                                                vocabulary_data=self.X_train)
            self.vocabulary_size = vect.vocabulary_size
            self.embedding_size = vect.embedding_size
            self.max_length = W2V_NUM_FEATURES
        elif self.feature_name is ONE_HOT:
            vect = OneHotTransformer(vocabulary_size=VOCABULARY_SIZE, max_length=MAX_LENGTH,
                                     stopwords=stop_words)
            self.vocabulary_size = VOCABULARY_SIZE
            self.embedding_size = EMBEDDING_SIZE
            self.max_length = MAX_LENGTH

        return {VECT: vect}

    def get_evaluation_params(self):
        optimizers = ['rmsprop', 'adam']
        epochs = [1, 5, 10]
        batch_sizes = [5, 10, 100]
        parameters = {'optimizer': optimizers, 'epochs': epochs, 'batch_size': batch_sizes}

        return self.add_clf_prefix(params=parameters)

    def save_model(self):
        # Save the Keras model first:
        self.model.named_steps[CLF].model.save(self.keras_model_path)
        self.model.named_steps[CLF].model = None

        # Save the rest pipeline
        super().save_model()

    def load_model(self):
        # Load the pipeline first:
        model = super().load_model()

        # Then, load the Keras model:
        keras_model = load_model(self.keras_model_path)
        model.named_steps[CLF].model = keras_model

        return model

    @staticmethod
    def get_fit_params():
        es = EarlyStopping(monitor='loss', patience=2, verbose=VERBOSE)
        callbacks = [es]
        return FakeNewsDeepLearner.add_clf_prefix(params={'callbacks': callbacks})

    def plot_model(self):
        dot_img_file = self.model_name + FORMAT_PNG
        plot_model(self.model.named_steps[CLF].model, to_file=dot_img_file, show_shapes=False, expand_nested=True)

