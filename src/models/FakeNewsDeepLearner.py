from keras_preprocessing.text import Tokenizer
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM
from tensorflow.keras.models import Sequential
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasRegressor

from src.models.SupervisedLearner import SupervisedLearner
from src.utils.conf import *


class FakeNewsDeepLearner(SupervisedLearner):

    def __init__(self, _learner_name, _feature_name, _evaluate, _df_test=None, _df_train=None):
        super().__init__(_learner_name, _feature_name, _evaluate, _df_test, _df_train)

    def create_pipeline(self, learner=None, features=None):

        if self.model_name is LSTM:
            # LSTM Neural Network
            model = Sequential(name='lstm_nn_model')
            model.add(layer=Embedding(input_dim=4500, output_dim=120, name='1st_layer'))
            model.add(layer=LSTM(units=120, dropout=0.2, recurrent_dropout=0.2, name='2nd_layer'))
            model.add(layer=Dropout(rate=0.5, name='3rd_layer'))
            model.add(layer=Dense(units=120, activation='relu', name='4th_layer'))
            model.add(layer=Dropout(rate=0.5, name='5th_layer'))
            model.add(layer=Dense(units=len(set(y)), activation='sigmoid', name='output_layer'))
        elif self.model_name is GRU:
            # GRU neural Network
            model = Sequential(name='gru_nn_model')
            model.add(layer=Embedding(input_dim=max_features, output_dim=120, name='1st_layer'))
            model.add(layer=GRU(units=120, dropout=0.2,
                                    recurrent_dropout=0.2, recurrent_activation='relu',
                                    activation='relu', name='2nd_layer'))
            model.add(layer=Dropout(rate=0.4, name='3rd_layer'))
            model.add(layer=Dense(units=120, activation='relu', name='4th_layer'))
            model.add(layer=Dropout(rate=0.2, name='5th_layer'))
            model.add(layer=Dense(units=len(set(y)), activation='softmax', name='output_layer'))
        else:
            # Default choice LSTM Neural Network
            model = Sequential(name='lstm_nn_model')
            model.add(layer=Embedding(input_dim=4500, output_dim=120, name='1st_layer'))
            model.add(layer=LSTM(units=120, dropout=0.2, recurrent_dropout=0.2, name='2nd_layer'))
            model.add(layer=Dropout(rate=0.5, name='3rd_layer'))
            model.add(layer=Dense(units=120, activation='relu', name='4th_layer'))
            model.add(layer=Dropout(rate=0.5, name='5th_layer'))
            model.add(layer=Dense(units=len(set(y)), activation='sigmoid', name='output_layer'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tokenizer = Tokenizer(num_words = 4500, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')

        # just create the pipeline
        pipeline = Pipeline([
            ('tokenizer', tokenizer),
            ('clf', KerasRegressor(model, verbose=0))
        ])

        return pipeline
