from copy import deepcopy

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import numpy as np
from sklearn.svm import SVC
import pandas as pd

import my_util as util


def train_regression_models(x_data, y_data, prep_pipelines, run_opts):
    print("----------------------------------")
    print("--- Training Regression Models ---")
    print("----------------------------------\n")

    # Creates pipelines for each product of feature selection and classifier (12 total)
    classifier_pipelines = create_classifier_pipelines(prep_pipelines)

    # Generates cross validation scores on pipelines
    cross_val_accuracy(classifier_pipelines, x_data, y_data)

    # Fits untuned versions of non-noisy pipelines to whole dataset
    fit_non_noisy_pipelines(classifier_pipelines, x_data, y_data)

    # Tunes hyperparameters for noisy pipelines
    tune_and_fit_noisy_pipelines(classifier_pipelines, x_data, y_data, run_opts)

    return classifier_pipelines


def gen_classifiers():
    return {
        'linear logistic regression (one vs rest)': LogisticRegression(penalty='l2', solver='liblinear',
                                                                      multi_class='ovr'),
        'linear svm (one vs one)': SVC(kernel='linear', probability=True),
        'rbf svm (one vs one)': SVC(kernel='rbf', probability=True),
        'k-neighbours': KNeighborsClassifier(n_neighbors=5, weights='uniform')
    }


def create_classifier_pipelines(prep_pipelines):
    print("Creating classifier pipelines...\n")
    classifier_pipelines = {}
    for ft_name, pl in prep_pipelines.items():
        classifier_pipelines[ft_name] = {}

        for cl_name, cl in gen_classifiers().items():  # See definition of gen_classifiers above
            # Must be copied to ensure pipelines are separate
            pl_copy = deepcopy(pl)
            pl_copy.steps.append([cl_name, cl])
            classifier_pipelines[ft_name][cl_name] = pl_copy
    return classifier_pipelines


# Generates cross validation scores on pipelines
def cross_val_accuracy(classifier_pipelines, x_data, y_data):
    print("Generating basic cross-validation scores for pipelines...\n")
    print("Test accuracy (mean across folds) | Pipeline: <feature type, classifier type>")
    for ft_name, feature_pipelines in classifier_pipelines.items():
        for cl_name, pipeline in feature_pipelines.items():
            cv_res = cross_validate(pipeline, x_data, np.ravel(y_data), scoring='accuracy', cv=3,
                                    return_train_score=False)
            print(round(cv_res['test_score'].mean() * 100, 1), "% | <", ft_name, ', ', cl_name, ">", sep='')


# Fits untuned versions of non-noisy pipelines to whole dataset
def fit_non_noisy_pipelines(classifier_pipelines, x_data, y_data):
    print("\nFitting non-noisy pipelines on training data...")
    for ft_name, feature_pipelines in classifier_pipelines.items():
        if ft_name == "noisy data classifier": continue
        for cl_name, pipeline in feature_pipelines.items():
            pipeline.fit(x_data, np.ravel(y_data))


# Uses cross-validation to tune parameters for the noisy pipelines, and then fits the estimators
# using the best parameter combinations. Plots the results of the grid search
def tune_and_fit_noisy_pipelines(classifier_pipelines, x_data, y_data, run_opts):
    print("\nTuning hyperparameters for noisy data...")

    noisy_pipelines = classifier_pipelines['noisy data classifier']
    # Turns the param grid into the correct naming format for grid search
    classifier_param_grids = {k: {k + '__' + kk: vv for kk, vv in v.items()} for k, v in classifier_params().items()}

    new_noisy_pipelines = {}
    for cl_name, pipeline in noisy_pipelines.items():
        # Uses cross-validation to find the best parameters for each noisy pipeline
        grid = GridSearchCV(pipeline, param_grid=classifier_param_grids[cl_name], scoring='neg_log_loss', cv=3,
                            refit='neg_log_loss', return_train_score=True)
        grid.fit(x_data, np.ravel(y_data))
        new_noisy_pipelines[cl_name] = grid  # Assigns the fitted pipeline to the dict of new pipelines

        # Plots the results of the grid search
        plot_grid_search_results(grid, cl_name, run_opts)

    classifier_pipelines['noisy data classifier'] = new_noisy_pipelines


def classifier_params():
    return {
        'linear logistic regression (one vs rest)': {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 1, 10]  # Inverse of regularization strength; smaller values specify stronger regularization.
        },
        'linear svm (one vs one)': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'rbf svm (one vs one)': {
            'C': [0.01, 1, 10],
            'gamma': [0.0001, 0.001]
        },
        'k-neighbours': {
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # p = 1: manhattan_distance, p = 2: euclidean_distance.
        }
    }
# classifier_param_grids = {
#     'linear logistic regression (one vs rest)': {
#         'penalty': ['l2'],
#         'C': [1]  # Inverse of regularization strength; smaller values specify stronger regularization.
#     },
#     'linear svm (one vs one)': {
#         'C': [1]
#     },
#     'rbf svm (one vs one)': {
#         'C': [1],
#         'gamma': [0.001]
#     },
#     'k-neighbours': {
#         'weights': ['distance'],
#         'p': [2]  # p = 1: manhattan_distance, p = 2: euclidean_distance.
#     }
# }


# Plots the results of the grid search
def plot_grid_search_results(grid, cl_name, run_opts):
    # Generates bar char labels
    score_mean_names = ['mean_train_score', 'mean_test_score']
    score_means = [grid.cv_results_[score_name] for score_name in score_mean_names]
    param_group_names = [to_axis_label(cl_name, d) for d in grid.cv_results_['params']]

    # Formats results in way that can be plotted
    results = pd.DataFrame(score_means, index=score_mean_names, columns=param_group_names)
    results *= -1
    results = results.transpose()

    # Plots chart
    ax = results.plot(kind='bar', title='cross-validation tuning: noisy ' + cl_name)
    ax.set_xlabel("parameter combinations")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylabel("log loss")

    # Sets best param combo to green
    best_label = ax.get_xticklabels()[results.index.get_loc(to_axis_label(cl_name, grid.best_params_))]
    best_label.set_color("white")
    best_label.set_bbox(dict(facecolor="green", alpha=0.9))
    best_label.set_weight("bold")

    util.plot('noisy ' + cl_name + ' cross-val log loss', run_opts, False)


# Creates axis labels from grid search params
def to_axis_label(cl_name, param_dict):

    def param_name_str(v):
        return 'γ' if v == 'gamma' else v

    return '\n'.join([param_name_str(util.remove_prefix(k, cl_name + '__')) + ': ' + str(v)
                      for k, v in param_dict.items()])
