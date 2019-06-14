from multiClassTask.analysis import analyse
from multiClassTask.evaluation import evaluate_performance
from load_and_clean_data import load_and_clean_data
from multiClassTask.model_training import train_regression_models
from multiClassTask.feature_preparation import select_and_prep_features
import my_util as util

# Reads in command line arguments and does other setup operations
run_opts = util.setup()

# Loads and cleans the input energy data from the CSV file
x_data, y_data, x_to_classify_data = load_and_clean_data('multiClassTask/data/')

# # Splits data into test and training
x_train, x_test, y_train, y_test = util.stratified_train_test_split(x_data, y_data, 0.2)

# Performs analysis of the data and creates visualisations
analyse(x_train.copy(), y_train.copy(), run_opts)

# Returns pipelines to prepare the inputs and choose a suitable subset of features
feature_pipelines = select_and_prep_features(x_data.columns.values)

# Selects and trains regression models on the training set, returning the best model
classifier_pipelines = train_regression_models(x_train, y_train, feature_pipelines, run_opts)

# Evaluates the performance of the model on the test data
best_pipeline = evaluate_performance(classifier_pipelines, x_test, y_test, run_opts)

# Uses the best pipeline from the evaluation to predict the classes for the data which needs classifying
util.to_file(best_pipeline.predict(x_to_classify_data), 'multiClassTask/PredictedClasses.csv')
