class ISupervisedLearner:
    def __init__(self, _model_name, _df_train=None, _df_test=None):
        pass

    def prepare_data(self, df_train=None, df_test=None):
        pass

    def prepare_test_data(self, df_test):
        pass

    def train_model(self, _model):
        pass

    def predict_proba(self, _df_test=None):
        pass

    def init_model(self):
        pass

    def load_trained_model(self, name):
        pass

    def create_pipeline(self):
        pass

    @staticmethod
    def prediction_to_csv(filename, test_id, y_test):
        pass

    @staticmethod
    def engineer_data(data_change, remove_outliers=False):
        pass

    @staticmethod
    def remove_outliers(data):
        pass

    @staticmethod
    def remove_useless_columns(data):
        pass

    @staticmethod
    def fill_na_values(data, na_value=' '):
        pass

    @staticmethod
    def remove_na_values(data):
        pass

    @staticmethod
    def create_features(data, times=5):
        pass
