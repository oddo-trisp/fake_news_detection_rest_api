# Utility methods

import os
import pickle
from pathlib import Path


def get_valid_path(destination):
    root_path = Path(__file__).parent.parent.parent
    working_dir = os.getcwd()
    steps = Path(working_dir).relative_to(root_path).as_posix().count('/') + 1

    prefix = ''
    for i in range(steps):
        prefix = prefix + '../'

    return Path(prefix + destination)


def save_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data



