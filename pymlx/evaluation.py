import random
import numpy
from pandas import DataFrame
from sklearn import metrics as skmetrics
from matplotlib import pyplot as plt
from .model import BinaryClassifier


def pr_scorer(predictor, X, y):
    p = predictor.predict_proba(X)[:, 1]
    precision, recall, thresholds = skmetrics.precision_recall_curve(y, p)
    return skmetrics.auc(recall, precision)


def log_loss(truths, predictions):
    return [- (truths[i] * numpy.log(p) + (1 - truths[i]) * numpy.log(1 - p))
            for i, p in enumerate(predictions)]


def _plot_curve(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.title('{0} Curve (AUC = {1:.4f})'.format(title, skmetrics.auc(x, y)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_roc_pr(truths, predictions):
    plt.rcParams['figure.figsize'] = 10, 4
    fpr, tpr, thresholds = skmetrics.roc_curve(truths, predictions)
    plt.subplot(1, 2, 1)
    _plot_curve(fpr, tpr, 'ROC', 'False positive rate', 'True positive rate')
    plt.subplot(1, 2, 2)
    precision, recall, thresholds = skmetrics.precision_recall_curve(truths, predictions)
    _plot_curve(recall, precision, 'PR', 'Recall', 'Precision')
    plt.tight_layout(w_pad=4)


def _show_confusion_matrix(truths, predictions, threshold):
    matrix = skmetrics.confusion_matrix(truths, [prob > threshold for prob in predictions])
    tn = matrix[0, 0]
    fp = matrix[0, 1]
    fn = matrix[1, 0]
    tp = matrix[1, 1]
    print('                 PREDICTED            Recall')
    print('         ======F=============T======')
    print('         |            |            |')
    print('         F{0:10}  |{1:10}  |  N: {2:.2f}%'.format(tn, fp, float(tn) / (tn + fp) * 100))
    print('         |            |            |')
    print('LABELED  |-------------------------|')
    print('         |            |            |')
    print('         T{0:10}  |{1:10}  |  P: {2:.2f}%'.format(fn, tp, float(tp) / (fn + tp) * 100))
    print('         |            |            |')
    print('         ===========================')
    print('Precision   N: {0:.2f}%     P: {1:.2f}%'.format(float(tn) / (tn + fn) * 100,
                                                           float(tp) / (fp + tp) * 100))


def confusion_matrix(truths, predictions, sampling_rate=1):
    from ipywidgets import interact, fixed
    assert len(truths) == len(predictions)
    if sampling_rate < 1:
        n = len(truths)
        indices = random.sample(range(n), int(n * sampling_rate))
        truths = [truths[i] for i in indices]
        predictions = [predictions[i] for i in indices]

    interact(_show_confusion_matrix,
             threshold=(0.0, 1.0, 0.01), truths=fixed(truths), predictions=fixed(predictions))


def diagnose(model, test_df, truths, cols=None, predictions=None, good_to_bad=False, pretty=True):
    assert isinstance(model, BinaryClassifier)
    test_features = model.featurizer.transform(test_df)
    if predictions is None:
        predictions = model.bulk_predict(test_features)

    df = DataFrame(test_features.todense(),
                   columns=model.featurizer.out_feature_names,
                   index=test_df.index)
    status_cols = ['Labeled', 'Predicted Probability', 'diff']
    for i, status in enumerate([truths, predictions,
                                [abs(truths[i] - p) for i, p in enumerate(predictions)]]):
        df[status_cols[i]] = status
    df = df[status_cols + list(cols)].sort_values('diff', ascending=good_to_bad)
    df.drop('diff', axis=1, inplace=True)
    if pretty:
        df.columns = [name.title().replace('_', ' ') for name in df.columns]
    return df
