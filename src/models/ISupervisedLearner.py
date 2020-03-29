class ISupervisedLearner:
    def __init__(self):
        pass

    def prepare_data(self, df_train=None, df_test=None):
        pass

    def prepare_test_data(self, df_test):
        pass

    def fit(self, _model):
        pass

    def predict_proba(self, _df_test=None):
        pass

    def load_model(self):
        pass

    @staticmethod
    def prediction_to_csv(filename, test_id, y_test):
        pass

    @staticmethod
    def engineer_data(data_change):
        pass

    @staticmethod
    def remove_outliers(data):
        pass

    @staticmethod
    def drop_useless_columns(data):
        pass

    @staticmethod
    def fill_na_values(data, na_value=' '):
        pass

    @staticmethod
    def create_features(data):
        pass

    @staticmethod
    def create_transformations(data, transformation=None, vectorizer=None):
        pass
