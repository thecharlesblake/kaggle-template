from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
import numpy as np
from copy import deepcopy
import my_util as util


def train_regression_models(x_data, y_data, prep_pipelines, run_opts):
    print("----------------------------------")
    print("--- Training Regression Models ---")
    print("----------------------------------\n")

    cross_val = 3
    train_score_fns = [
        # 'accuracy',
        # 'precision',
        # 'recall',
        'f1',
        'roc_auc'
    ]

    print("Adding logistic regression step to pipelines...\n")
    trained_pipelines = {n: make_pipeline(p, LogisticRegression(
        penalty='l2',
        C=1,
        solver='liblinear',
        multi_class='ovr'
    )) for n, p in prep_pipelines.items()}

    for p_name, pipeline in trained_pipelines.items():
        print("Generating cross-validation scores for pipeline", p_name, "...")
        cv_res = cross_validate(pipeline, x_data, np.ravel(y_data), scoring=train_score_fns, cv=cross_val, return_train_score=False)
        cv_res = {'mean_'+k: v.mean() for k, v in cv_res.items()}
        print(cv_res)
        print("Fitting pipeline", p_name, "on training data...\n")
        pipeline.fit(x_data, np.ravel(y_data))

    return trained_pipelines
