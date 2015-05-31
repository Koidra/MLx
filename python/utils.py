import csv
from numpy import log
from pandas import read_csv, DataFrame, Series
from sklearn import metrics as skmetrics
from sklearn.grid_search import RandomizedSearchCV
from matplotlib import pyplot as plt

def load_data(filename, label_col, id_col=None, feature_cols=None, excluded_cols=None):
    if feature_cols is not None and excluded_cols is not None:
        raise ValueError('Either feature_cols or excluded_columns must be None.')
    included_columns = None if feature_cols is None else ([label_col] + list(feature_cols))
    if included_columns and id_col:
        included_columns.append(id_col)

    df = read_csv(filename, index_col=id_col, usecols=included_columns)
    labels = list(df[label_col])
    df.drop([label_col] + (excluded_cols if excluded_cols else []), axis=1, inplace=True)
    return df, labels

def random_sweep(X, y, model, params, n_iter, scoring='roc_auc', cv=3,
                 refit=False, n_jobs=1, verbose=2):
    sweeper = RandomizedSearchCV(model, params, scoring=scoring, n_iter=n_iter,
                                 cv=cv, refit=refit, n_jobs=n_jobs, verbose=verbose)
    sweeper.fit(X, y)
    return sweeper

def sweep_stats(sweeper, low_is_good=False):
    stats = []
    for row in sorted(sweeper.grid_scores_,
                      key=lambda x: x.mean_validation_score, reverse=low_is_good):
        stat = {'Score': row.mean_validation_score, 'Std': row.cv_validation_scores.std()}
        parameters = row.parameters
        for param in parameters:
            stat[param] = parameters[param]
        stats.append(stat)
    return DataFrame(stats)

def get_fscores(predictor, feature_names=None):
        fscores = predictor.booster().get_fscore()
        if feature_names is not None:
            fscores = {feature_names[int(f[1:])]: fscores[f] for f in fscores}
        return Series(fscores).order(ascending=False)

def log_loss(truths, predictions):
    return [- (truths[i] * log(p) + (1 - truths[i]) * log(1 - p))
            for i, p in enumerate(predictions)]

def diagnose(model, test_df, truths, cols=None, predictions=None, good_to_bad=False, pretty=True):
    import MLx
    assert isinstance(model, MLx.BinaryClassifier)
    test_features = model.featurizer.transform(test_df)
    if predictions is None:
        predictions = model.bulk_predict(test_features)

    df = DataFrame(test_features.todense(),
                   columns=model.featurizer.out_feature_names,
                   index=test_df.index)
    status_cols = ['Truth', 'Predicted Probability', 'Log Loss']
    for i, status in enumerate([truths, predictions, log_loss(truths, predictions)]):
        df[status_cols[i]] = status
    df = df[status_cols + list(cols)].sort('Log Loss', ascending=good_to_bad)
    if pretty:
        df.columns = [name.title().replace('_', ' ') for name in df.columns]
    return df

def plot_roc(truths, predictions):
    fpr, tpr, thresholds = skmetrics.roc_curve(truths, predictions)
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    print('ROC AUC: {0:.2f}'.format(skmetrics.auc(fpr, tpr)))

def _show_confusion_matrix(truths, predictions, threshold):
    matrix = skmetrics.confusion_matrix(truths, [prob > threshold for prob in predictions])
    tn = matrix[0,0]
    fp = matrix[0,1]
    fn = matrix[1,0]
    tp = matrix[1,1]
    print('                 PREDICTED            Recall')
    print('         ======F=============T======')
    print('         |            |            |')
    print('         F{0:10}  |{1:10}  |  N: {2:.2f}%'.format(tn, fp, float(tn)/(tn+fp)*100))
    print('         |            |            |')
    print('TRUTH    |-------------------------|')
    print('         |            |            |')
    print('         T{0:10}  |{1:10}  |  P: {2:.2f}%'.format(fn, tp, float(tp)/(fn+tp)*100))
    print('         |            |            |')
    print('         ===========================')
    print('Precision   N: {0:.2f}%     P: {1:.2f}%'.format(float(tn)/(tn + fn)*100, float(tp)/(fp+tp)*100))

def confusion_matrix(truths, predictions):
    from IPython.html.widgets import interact, fixed
    interact(_show_confusion_matrix,
             Threshold=(0.0, 1.0, 0.05), truths=fixed(truths), predictions=fixed(predictions))


def histogram(filename, cols):
    if isinstance(cols, list):
        with open(filename, 'rb') as f:
            reader = csv.reader(f)
            header = reader.next()
            col_idx = {col: header.index(col) for col in cols}
            histograms = {col: {} for col in cols}
            for row in reader:
                for col in cols:
                    histogram = histograms[col]
                    value = row[col_idx[col]]
                    if value in histogram:
                        histogram[value] += 1
                    else:
                        histogram[value] = 1
        for col in cols:
            hist = histograms[col]
            print(col + ':')
            for pair in sorted(hist.items(), key=lambda x: x[1], reverse=True):
                print(' {0}\t{1}'.format(pair[0], pair[1]))
            print
    else:
        col = cols
        with open(filename, 'rb') as f:
            reader = csv.reader(f)
            col_id = reader.next().index(col)
            hist = {}
            for row in reader:
                value = row[col_id]
                if value in hist:
                    hist[value] += 1
                else:
                    hist[value] = 1
        for pair in sorted(hist.items(), key=lambda x: x[1], reverse=True):
            print(' {0}\t{1}'.format(pair[0], pair[1]))