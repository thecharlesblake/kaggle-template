import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, average_precision_score, \
    roc_curve
import my_util as util


def evaluate_performance(pipelines, x_data, y_data, run_opts):
    print("------------------------")
    print("--- Model Evaluation ---")
    print("------------------------\n")

    y_true = np.ravel(y_data)
    total = len(y_true)

    print("Evaluating test data using trained pipelines...\n")

    y_probas = []
    for p_name, pipeline in pipelines.items():
        y_pred = pipeline.predict(x_data)
        y_score = pipeline.decision_function(x_data)
        y_probas.append(pipeline.predict_proba(x_data)[:, 1])

        correct = 0
        for t, p in zip(y_true, y_pred):
            if t == p:
                correct += 1

        print("--- Evaluation for pipeline:", p_name, "---\n")

        print(correct, "correct, out of", total, "total")

        print("Accuracy of ", p_name, " pipeline: ", round(accuracy_score(y_true, y_pred)*100, 2), '%', sep='')

        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=['green', 'red']))

        average_precision = average_precision_score(y_true, y_score)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))

        print("Generating precision-recall graph...\n")
        plot_precision_recall(y_true, y_score, average_precision, p_name, run_opts)

    plot_roc_curve(y_true, y_probas, pipelines.keys(), run_opts)

    return pipelines['653.930 nm classifier']


# Based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html example
def plot_precision_recall(y_true, y_score, average_precision, p_name, run_opts):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.step(recall, precision, color='g', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.title(p_name+' precision-recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    util.plot(p_name+'_precision_recall', run_opts, True)


def plot_roc_curve(y_true, y_probas, p_names, run_opts):
    for y_proba, p_name, linestyle, col in zip(y_probas, p_names, ['--',':','-'], ['grey','r','b']):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, linestyle, label=p_name, alpha=0.5, c=col, lw=3)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    util.plot('roc_curve', run_opts, True)
