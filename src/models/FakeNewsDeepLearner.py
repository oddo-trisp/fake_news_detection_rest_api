from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from src.models.SupervisedLearner import SupervisedLearner
from src.models.transformers import AverageWordVectorTransformer, PadSequencesTransformer
from src.utils.conf import *


class FakeNewsDeepLearner(SupervisedLearner):

    def __init__(self, learner_name, feature_name, evaluate, language=ENGLISH, df_train=None, df_test=None):
        super().__init__(learner_name, feature_name, evaluate, language, df_train, df_test)
        self.vocabulary_size = None
        self.embedding_size = None
        self.max_length = None

    # TODO: Add optimal parameters from grid search
    def create_learner(self):
        nn_clf = KerasClassifier(build_fn=self.create_nn_model, epochs=10, batch_size=64, verbose=1)
        return {'clf': nn_clf}

    def create_nn_model(self):
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

        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(nn_model.summary())
        return nn_model

    def create_default_learner(self):
        pass

    def create_features(self):
        vect = None

        stop_words = self.get_stopwords()

        if self.feature_name is W2V:
            vect = AverageWordVectorTransformer(language=self.language, stop_words=stop_words,
                                                vocabulary_data=self.X_train)
            self.vocabulary_size = vect.vocabulary_size
            self.embedding_size = vect.embedding_size
            self.max_length = W2V_NUM_FEATURES
        elif self.feature_name is PAD_SEQ:
            vect = PadSequencesTransformer(vocabulary_size=VOCABULARY_SIZE, max_length=MAX_LENGTH,
                                           stopwords=stop_words)
            self.vocabulary_size = VOCABULARY_SIZE
            self.embedding_size = EMBEDDING_SIZE
            self.max_length = MAX_LENGTH

        return {'vect': vect}

    @staticmethod
    def get_fit_params():
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
        callbacks = [es]
        return {'callbacks': callbacks}
