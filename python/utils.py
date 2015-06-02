import csv
import random
import pandas
from pandas import DataFrame, Series, read_csv
from sklearn import metrics as skmetrics
from sklearn.grid_search import RandomizedSearchCV
from matplotlib import pyplot as plt
from IPython.display import display
from features_handler import *
from featurizer import Featurizer
from composition_handler import CompositionHandler

#### Core Utils ####
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

##### Data Ingestion #####
def load_data(filename, label_col, id_col=None, feature_cols=None, excluded_cols=None, nrows=None):
    if feature_cols is not None and excluded_cols is not None:
        raise ValueError('Either feature_cols or excluded_columns must be None.')
    included_columns = None if feature_cols is None else ([label_col] + list(feature_cols))
    if included_columns and id_col:
        included_columns.append(id_col)

    df = read_csv(filename, index_col=id_col, usecols=included_columns, nrows=nrows)
    labels = list(df[label_col])
    df.drop([label_col] + (excluded_cols if excluded_cols else []), axis=1, inplace=True)
    return df, labels


##### Exploration #####

def suggest_handlers(df, sample_size = 10000, trees_optimized=True, hinted_featurizer=None):
    """
    Suggest handlers given the data
    :param df: DataFrame
    :param trees_optimized: if true, the handlers are suggested to optimize trees learning
     Otherwise, they are suggested for other learners (e.g. linear or NN)
    """
    if len(df) > sample_size:
        df = df.ix[random.sample(df.index, sample_size)]

    CAT_INT = 'NEED REVIEW: could be either CategoricalHandler or a numeric handler'
    NA = 'Too difficult to tell'
    numeric_kinds = [BoolHandler, BinNormalizer, 'MinMaxNormalizer', NoHandler]
    kinds = [CategoricalHandler, 'TextHandler'] + numeric_kinds + \
            [CAT_INT, MapperHandler, CompositionHandler, PredicatesHandler, NA]

    handlers = {kind: [] for kind in kinds}
    priors = set()
    if hinted_featurizer:
        assert isinstance(hinted_featurizer, Featurizer)
        for handler in hinted_featurizer.handlers:
            for col in handler.in_feature_names:
                kind = handler.__class__
                if col not in handlers[kind]:
                    handlers[kind].append(col)
                    priors.add(col)
                    if kind in [CategoricalHandler, BoolHandler] and df[col].dtype != object:
                        df[[col]] = df[[col]].astype(object)

    for col in df:
        if col in priors:
            continue

        dtype = Counter(type(value) for value in df[col]
                        if not (value != value)).most_common()[0][0]
        if dtype == bool:
            kind = BoolHandler
            if df[col].dtype != object:
                # this is a hack because df.describe doesn't work correctly for bool series
                # https://github.com/pydata/pandas/issues/6625
                df[[col]] = df[[col]].astype(object)
        elif dtype == str:
            kind = CategoricalHandler
            # ToDo: could also be TEXT
        elif numpy.issubdtype(dtype, numpy.integer):
            col_low = col.lower()
            if col_low.endswith('id') or col_low.endswith('code') or col_low.endswith('type'):
                kind = CategoricalHandler
                df[[col]] = df[[col]].astype(str)
            elif col_low.endswith('count') or col_low.endswith('s'):
                kind = 'numeric'
            else:
                kind = CAT_INT
        elif numpy.issubdtype(dtype, numpy.float):
            kind = 'numeric'
        else:
            kind = NA

        if kind == 'numeric':
            if trees_optimized:
                kind = BinNormalizer if len(set(df[col])) > 32 else NoHandler
            else:
                raise ValueError('Not yet implemented')
        handlers[kind].append(col)

    ret = AttrDict()
    pandas.options.display.float_format = '{:.2g}'.format
    pandas.options.display.max_rows = 500
    for kind in kinds:
        if len(handlers[kind]) == 0:
            del handlers[kind]
        else:
            cols = sorted(handlers[kind])
            df_sub = df[cols]
            if kind == CategoricalHandler or kind == BoolHandler:
                desc = df_sub.describe(include=[object]).transpose()
            elif kind in numeric_kinds:
                desc = df_sub.describe(include=[numpy.number]).transpose()
            else:
                desc = DataFrame(index=cols)

            try:
                kind = kind.__name__
            except AttributeError:
                pass
            print(kind)
            if set(desc.index) != set(cols):
                print('Outlier columns: {0}'.format(sorted(set(cols) - set(desc.index))))
            samples = df_sub.ix[random.sample(df_sub.index, 5)]
            desc['Sample values'] = Series(
                [', '.join(('{:.2g}' if isinstance(val, float) else '{:}')
                           .format(val) for val in samples[col])
                 for col in samples], index=cols)
            desc.columns = [col.title() for col in desc.columns]
            ret[kind] = desc

            display(desc)
            print

    return ret


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
    return [- (truths[i] * numpy.log(p) + (1 - truths[i]) * numpy.log(1 - p))
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

def _plot_curve(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.title('{0} Curve (AUC = {1:.2f})'.format(title, skmetrics.auc(x, y)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def plot_roc_pr(truths, predictions):
    plt.rcParams['figure.figsize'] = 10, 4
    fpr, tpr, thresholds = skmetrics.roc_curve(truths, predictions)
    plt.subplot(1,2,1)
    _plot_curve(fpr, tpr, 'ROC', 'False positive rate', 'True positive rate')
    plt.subplot(1,2,2)
    precision, recall, thresholds = skmetrics.precision_recall_curve(truths, predictions)
    precision, recall = zip(*sorted(zip(precision, recall)))
    _plot_curve(precision, recall, 'PR', 'Precision', 'Recall')
    plt.tight_layout(w_pad=4)

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
             threshold=(0.0, 1.0, 0.05), truths=fixed(truths), predictions=fixed(predictions))

def histogram(filename, cols):
    if isinstance(cols, list):
        with open(filename, 'rb') as f:
            reader = csv.reader(f)
            header = reader.next()
            col_idx = {col: header.index(col) for col in cols}
            histograms = {col: {} for col in cols}
            for row in reader:
                for col in cols:
                    hist = histograms[col]
                    value = row[col_idx[col]]
                    if value in hist:
                        hist[value] += 1
                    else:
                        hist[value] = 1
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

def field_types(df):
    types = {}
    for col in df:
        t = str(df[col].dtype)
        if t in types:
            types[t].append(col)
        else:
            types[t] = [col]
    return types