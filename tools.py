import warnings

import numpy as np
import pandas as pd
import tensorflow as tf


def split_dataset(dataset, test_ratio=0.1):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def set_env_variables(seed):
    warnings.filterwarnings(action="ignore")

    pd.options.display.max_seq_items = 8000
    pd.options.display.max_rows = 8000

    np.random.seed(seed)
    tf.random.set_seed(seed)

    if not tf.executing_eagerly():
        tf.compat.v1.reset_default_graph()
