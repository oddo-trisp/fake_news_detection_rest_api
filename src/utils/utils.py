# Utility methods

import os
import pickle
import pandas as pd
from pathlib import Path


def get_valid_path(destination):
    root_path = Path(__file__).parent.parent.parent
    working_dir = os.getcwd()
    steps = Path(working_dir).relative_to(root_path).as_posix().count('/') + 1

    prefix = ''
    for i in range(steps):
        prefix = prefix + '../'

    return Path(prefix + destination)


def save_csv_file(file_path, data, index=None, columns=None, id_column=None, float_format=None, sep=','):
    df = pd.DataFrame(data, index=index, columns=columns)

    if id_column is not None:
        df['id'] = id_column

    df.to_csv(file_path, sep=sep, float_format=float_format)


def read_csv(file_path):
    return pd.read_csv(file_path)


def create_dataframe(data, columns=None):
    pd.DataFrame(data, columns=columns)


def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
