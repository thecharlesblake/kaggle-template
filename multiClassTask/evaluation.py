from itertools import product

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import my_util as util
import pandas as pd


def evaluate_performance(pipelines, x_data, y_data, run_opts):
    print("------------------------")
    print("--- Model Evaluation ---")
    print("------------------------\n")

    colours = ['blue', 'green', 'pink', 'red', 'gold']
    y_true = np.ravel(y_data)

    print("Evaluating test data using trained pipelines...")

    evaluate_scores(pipelines, x_data, y_true)

    # Plots graphs of decision regions for each high-low mean pipeline
    plot_decision_regions(pipelines, x_data, y_data, run_opts)

    # Plots confusion matrices for noisy pipelines
    plot_noisy_conf_matrices(pipelines, x_data, y_data, colours, run_opts)

    return pipelines['selected wavelength classifier']['rbf svm (one vs one)']


def evaluate_scores(pipelines, x_data, y_true):
    eval_results = {}
    total = len(y_true)

    # Generates for each pipeline, pred, score & proba
    for ft_name, feature_pipelines in pipelines.items():
        eval_results[ft_name] = {}
        for cl_name, pipeline in feature_pipelines.items():
            name = "(" + ft_name + ", " + cl_name + ")"

            y_pred = pipeline.predict(x_data)
            y_score = None if cl_name == 'k-neighbours' else pipeline.decision_function(x_data)
            y_proba = pipeline.predict_proba(x_data)[:, 1]
            eval_results[ft_name][cl_name] = {'y_pred': y_pred, 'y_score': y_score, 'y_proba': y_proba}

            correct = 0
            for t, p in zip(y_true, y_pred):
                if t == p:
                    correct += 1

            print("\n--- Evaluation for pipeline:", name, "---\n")

            print(correct, "correct, out of", total, "total")

            print("Accuracy of ", name, " pipeline: ", round(accuracy_score(y_true, y_pred) * 100, 2), '%', sep='')

            print("Classification report:")
            print(classification_report(y_true, y_pred, target_names=['blue', 'green', 'pink', 'red', 'yellow']))


def plot_decision_regions(pipelines, x_data, y_data, run_opts):
    xx, yy = np.meshgrid(np.arange(-10, 70, 0.1),
                         np.arange(-10, 70, 0.1))

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    # trick to add title and axis labels to whole chart
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title("Decision boundaries for low-high wavelength classifiers", y=1.08)
    plt.xlabel("mean intensity of wavelengths 550-600nm")
    plt.ylabel("mean intensity of wavelengths 600-650nm")

    cmap_light = ListedColormap(['#42A5F5', '#66BB6A', '#F48FB1', '#EF5350', '#FFCA28'])
    cmap_dark = ListedColormap(['#0D47A1', '#1B5E20', '#F06292', '#B71C1C', '#FFA000'])

    # Plots decision region charts in all 4 corners
    for idx, (cl_name, pipeline) in zip(product([0, 1], [0, 1]), pipelines['low-high means classifier'].items()):

        # Mesh to create the regions
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points = pipeline.named_steps.standardscaler.transform(mesh_points)
        Z = pipeline.named_steps[cl_name].predict(mesh_points)
        Z = Z.reshape(xx.shape)

        # Plots the mesh
        axarr[idx[0], idx[1]].pcolormesh(xx, yy, Z, cmap=cmap_light)
        low_x_means = pipeline.named_steps.dataframemeanselector.transform(x_data).iloc[:, 0]
        high_x_means = pipeline.named_steps.dataframemeanselector.transform(x_data).iloc[:, 1]

        # Plots the data points
        axarr[idx[0], idx[1]].scatter(low_x_means, high_x_means, c=y_data['colour'], cmap=cmap_dark, s=20, edgecolor='k')

        axarr[idx[0], idx[1]].set_title(cl_name)

    util.plot('low-high decision bounds', run_opts, False)


def plot_noisy_conf_matrices(pipelines, x_data, y_data, colours, run_opts):
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    # trick to add title and axis labels to whole chart
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title("Confusion matrices for noisy data classifiers", y=1.08)
    plt.xlabel("predicted label")
    plt.ylabel("true label")

    for idx, (cl_name, pipeline), cmap in zip(
            product([0, 1], [0, 1]),
            pipelines['noisy data classifier'].items(),
            [plt.cm.Blues, plt.cm.Reds, plt.cm.Oranges, plt.cm.Greens]):
        cnf_matrix = confusion_matrix(y_data, pipeline.predict(x_data))
        plot_confusion_matrix(cnf_matrix, axarr[idx[0], idx[1]], cl_name, colours, cmap)
    util.plot('noisy confusion matrices', run_opts, False)


def plot_confusion_matrix(cm, ax, title, classes, cmap):
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_title(title)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")
