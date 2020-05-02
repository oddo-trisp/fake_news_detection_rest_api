class ISupervisedLearner:

    def prepare_data(self, df_train=None, df_test=None):
        pass

    def prepare_train_data(self, df_train):
        pass

    def prepare_test_data(self, df_test):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass

    def hyperparameters_evaluation(self):
        pass

    def k_fold_evaluation(self, model, metrics):
        pass

    def simple_train_model(self):
        pass

    def predict_proba(self, df_test=None):
        pass

    def init_model(self):
        pass

    def load_trained_model(self):
        pass

    def create_pipeline(self, learner=None, features=None):
        pass

    def create_learner(self):
        pass

    def create_default_learner(self):
        pass

    def create_features(self):
        pass

    @staticmethod
    def get_k_fold():
        pass

    @staticmethod
    def get_stopwords(language):
        pass

    @staticmethod
    def save_prediction_to_csv(filename, test_id, y_test):
        pass

    @staticmethod
    def engineer_data(data, remove_outliers=False):
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
    def create_new_columns(data, times=5):
        pass

    @staticmethod
    def plot_roc_curve(metrics):
        pass

    @staticmethod
    def save_metrics_to_csv(metrics):
        pass

    @staticmethod
    def rename_labels(metrics, label, proper_label):
        pass
