from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM
from tensorflow.keras.models import Sequential
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasRegressor

from src.models.SupervisedLearner import SupervisedLearner
from src.utils.conf import *


class FakeNewsDeepLearner(SupervisedLearner):

    def __init__(self, learner_name, feature_name, evaluate, language=ENGLISH, df_train=None, df_test=None):
        super().__init__(learner_name, feature_name, evaluate, language, df_train, df_test)

    # TODO: Add optimal parameters from grid search
    def create_learner(self):
        nn_model = None

        if self.model_name is LSTM:
            # LSTM Neural Network
            nn_model = Sequential(name='lstm_nn_model')
            nn_model.add(layer=Embedding(input_dim=4500, output_dim=120, name='1st_layer'))
            nn_model.add(layer=LSTM(units=120, dropout=0.2, recurrent_dropout=0.2, name='2nd_layer'))
            nn_model.add(layer=Dropout(rate=0.5, name='3rd_layer'))
            nn_model.add(layer=Dense(unitsv=120, activation='relu', name='4th_layer'))
            nn_model.add(layer=Dropout(rate=0.5, name='5th_layer'))
            nn_model.add(layer=Dense(units=len(set(self.y_train)), activation='sigmoid', name='output_layer'))
        elif self.model_name is GRU:
            # GRU neural Network
            nn_model = Sequential(name='gru_nn_model')
            nn_model.add(layer=Embedding(input_dim=4500, output_dim=120, name='1st_layer'))
            nn_model.add(layer=GRU(units=120, dropout=0.2,
                                   recurrent_dropout=0.2, recurrent_activation='relu',
                                   activation='relu', name='2nd_layer'))
            nn_model.add(layer=Dropout(rate=0.4, name='3rd_layer'))
            nn_model.add(layer=Dense(units=120, activation='relu', name='4th_layer'))
            nn_model.add(layer=Dropout(rate=0.2, name='5th_layer'))
            nn_model.add(layer=Dense(units=len(set(self.y_train)), activation='softmax', name='output_layer'))

        nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        learner = KerasRegressor(nn_model, verbose=0)

        return learner

    def create_default_learner(self):
        pass
