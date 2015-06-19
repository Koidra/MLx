import csv
from time import time
from pandas import Series, read_csv
from .features_handler import *
from .featurizer import Featurizer

# Data Ingestion Utils #
def load_data(filename, label_col=None, id_col=None, feature_cols=None, excluded_cols=None,
              dtype=None):
    if feature_cols is not None and excluded_cols is not None:
        raise ValueError('Either feature_cols or excluded_columns must be None.')
    if feature_cols is None:
        included_columns = None
    elif label_col is None:
        included_columns = feature_cols
    else:
        included_columns = [label_col] + list(feature_cols)
    if included_columns and id_col:
        included_columns.append(id_col)

    df = read_csv(filename, index_col=id_col, usecols=included_columns, dtype=dtype)
    excluded_cols = excluded_cols or []
    if label_col:
        labels = list(df[label_col])
        df.drop([label_col] + excluded_cols, axis=1, inplace=True)
        return df, labels
    else:
        df.drop(excluded_cols, axis=1, inplace=True)
        return df


def load_featurized_data(filename, label_col, feat, in_memory=True):
    """
    Load raw data into extracted features and labels
    Note that streaming featurization is much slower than in-memory featurization,
     perhaps due to Pandas' read_csv being much faster than csv reader
     http://softwarerecs.stackexchange.com/questions/7463/fastest-python-library-to-read-a-csv-file
    """
    assert isinstance(feat, Featurizer)
    if in_memory:
        f_names = feat.in_feature_names
        f_types = feat.in_feature_types
        df, labels = load_data(filename, label_col,
                               feature_cols=f_names,
                               dtype={name: f_types[i] for i, name in enumerate(f_names)})
        return feat.transform(df), labels
    else:
        dtypes = feat.in_feature_types
        dim = len(dtypes)
        with open(filename, 'rb') as csv_file:
            reader = csv.reader(csv_file)
            header = reader.next()
            label_index = header.index(label_col)
            feature_indices = [header.index(f) for f in feat.in_feature_names]

            data = feat.init_data()
            labels = []
            for row in reader:
                raw_features = [0] * dim
                for i, idx in enumerate(feature_indices):
                    raw = row[idx]
                    raw_features[i] = dtypes[i](raw) \
                        if (raw or not numpy.issubdtype(dtypes[i], numpy.number)) else 0
                feat.add_features(data, raw_features)
                labels.append(numpy.float32(row[label_index]))

        return feat.to_matrix(data), labels


def get_fscores(predictor, feature_names=None):
    fscores = predictor.booster().get_fscore()
    if feature_names is not None:
        fscores = {feature_names[int(f[1:])]: fscores[f] for f in fscores}
    return Series(fscores).order(ascending=False)


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


_start_time = None
_elapsed = None
_time_unit_conversion = {'hour': 3600, 'minute': 60, 'second': 1}


def start_timing():
    global _start_time
    _start_time = time()


def report_timing(unit='minute', fresh=True):
    global _elapsed
    if fresh:
        _elapsed = time() - _start_time
    print('Elapsed: {0:.2g} {1}s'.format(_elapsed / _time_unit_conversion[unit], unit))
