import os
import matplotlib.pyplot as plt
import argparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sys import version_info
import numpy as np
import pandas as pd


IMAGES_PATH = "visualisations"

# NOTE: save_fig taken from the Hands-On Machine Learning Course Textbook
def save_fig(fig_id, binary, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if (binary):
        path = os.path.join('binaryTask', IMAGES_PATH, fig_id + "." + fig_extension)
    else:
        path = os.path.join('multiClassTask', IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def stratified_train_test_split(x, y, test_sz):
    print("Splitting dataset into train and test set (stratified)...\n")
    return train_test_split(x, y, stratify=y, test_size=test_sz)


def plot(plot_name, run_opts, binary):
    if run_opts.save_graphs:
        save_fig(plot_name, binary)
        plt.close()
    if run_opts.show_graphs:
        plt.show()
    if not run_opts.save_graphs or run_opts.show_graphs:
        plt.close()
    pass


def to_file(colour_vals, filename):
    with open(filename, 'w') as f:
        for val in colour_vals:
            print(val, file=f)


def setup():
    if version_info[0] < 3:
        raise Exception("Python 3 or later is required")

    np.random.seed(0)

    # Parses command line arguments
    parser = argparse.ArgumentParser(description='Run my classification project code.')
    parser.add_argument('-sh, --show-graphs', dest='show_graphs', action='store_true', default=False,
                        help='shows the generated graphs in pop-up windows to the user')
    parser.add_argument('-sv, --save-graphs', dest='save_graphs', action='store_true', default=False,
                        help='saves the generated graphs as images, in the "visualisations/" directory')

    args = parser.parse_args()
    return args


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text



### CODE TAKEN FROM HANDS-ON MACHINE LEARNING BOOK ###

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


class DataFrameMeanSelector(BaseEstimator, TransformerMixin):
    def __init__(self, ranges):
        self.ranges = ranges
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_split_means = {name: X.loc[:, lower:upper].mean(axis=1) for name, (lower, upper) in self.ranges.items()}
        return pd.DataFrame(X_split_means)


class DataFrameNoiseGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        noise = np.random.normal(0, 100, X.shape)
        return X + noise
