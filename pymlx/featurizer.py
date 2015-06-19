import random
import numpy as np
import pandas
from pandas import Series, read_csv
from scipy.sparse import csr_matrix
from .core import *
from .features_handler import *


class Featurizer:
    def __init__(self, handlers, sparse=True):
        assert isinstance(handlers, list)
        in_feature_names = set()
        for handler in handlers:
            assert isinstance(handler, FeaturesHandler)
            in_feature_names |= set(handler.in_feature_names)
        self.handlers = handlers
        self.in_feature_names = sorted(in_feature_names)
        self.in_feature_types = None
        self.out_feature_names = None
        self._handlers_feature_indices = None

        dtype = np.float32
        if sparse:
            self._features_ctor = lambda: ([], [])  # ctor for a features vector
            self.init_data = lambda: ([], [], [0])  # ctor for the features matrix
            self.to_matrix = lambda X: csr_matrix(X, shape=(len(X[2]) - 1, self.size()),
                                                  dtype=dtype)

            def add_feature(features, i, v):  # add one feature to a feature vector
                features[0].append(i)
                features[1].append(v)

            def add_features(data, features_vec):  # add a feature vector to a matrix
                data[0].extend(features_vec[1])  # data
                data[1].extend(features_vec[0])  # indices
                data[2].append(len(data[0]))  # ind_ptr

        else:
            self._features_ctor = None  # for dense, need size() first
            self.init_data = lambda: []
            self.to_matrix = lambda X: np.array(X, dtype=dtype)

            def add_feature(features, i, v):
                features[i] = v

            def add_features(data, features_vec):
                data.append(features_vec)

        self._add_feature = add_feature
        self._add_features = add_features

    def size(self):
        return len(self.out_feature_names)

    # Contracts: df must contain only feature columns
    def learn(self, data, sample_size=None):
        if isinstance(data, str):
            df = read_csv(data, usecols=self.in_feature_names)
        else:
            assert isinstance(data, DataFrame)
            df = data

        if sample_size and len(df) > sample_size:
            df = df.ix[random.sample(list(df.index), sample_size)]

        # REVIEW: read_csv() doesn't honor the order of the input column names
        # So we need to re-order in_feature_names to make it consistent with the data order
        self.in_feature_names = list(df.columns)
        self.in_feature_types = []
        name_to_index = {}
        for i, col in enumerate(df):
            dtype = Counter(type(value)
                            for value in df[col] if (value == value)).most_common(1)[0][0]
            self.in_feature_types.append(np.float32 if dtype == np.float64
                                         else np.int32 if dtype == np.int64 else dtype)
            name_to_index[col] = i

        self.out_feature_names = []
        self._handlers_feature_indices = []  # indices of the input features for each handler

        for handler in self.handlers:
            handler.learn(df)
            self._handlers_feature_indices.append([name_to_index[name]
                                                   for name in handler.in_feature_names])
            self.out_feature_names.extend(handler.out_feature_names)

        self._features_ctor = self._features_ctor or (lambda: [0] * self.size())

    # Note that we pass through NaNs
    # So either the handlers or the learner needs to handle NaNs
    # The featurizer itself doesn't handle NaNs
    def extract_features(self, raw_features):
        features = self._features_ctor()
        add_feature = self._add_feature
        offset = 0
        for h, handler in enumerate(self.handlers):
            feature_indices = self._handlers_feature_indices[h]
            for index, value in handler.apply(raw_features[i] for i in feature_indices):
                if value:
                    add_feature(features, offset + index, value)
            offset += handler.size()
        return features

    def add_features(self, data, raw_features):
        self._add_features(data, self.extract_features(raw_features))

    # Note: df cannot contain the label column
    def transform(self, df):
        assert isinstance(df, DataFrame)
        data = self.init_data()
        for row in df.itertuples(index=False):
            self.add_features(data, row)
        return self.to_matrix(data)


class CompositionHandler(FeaturesHandler):
    def __init__(self, handler1, handler2):
        assert isinstance(handler2, FeaturesHandler)
        if isinstance(handler1, list):
            for handler in handler1:
                assert isinstance(handler, FeaturesHandler)
        else:
            assert isinstance(handler1, FeaturesHandler)
            handler1 = [handler1]
        self._preprocessor = Featurizer(handler1, sparse=False)
        self._postprocessor = handler2
        self._intermediate_dimension = len(handler2.in_feature_names)
        self.in_feature_names = sorted(set().union(*[handler.in_feature_names
                                                     for handler in self._preprocessor.handlers]))

    def learn(self, df):
        preprocessor = self._preprocessor
        postprocessor = self._postprocessor
        df_sub = df[self.in_feature_names]
        preprocessor.learn(df_sub)
        assert preprocessor.out_feature_names == postprocessor.in_feature_names
        intermediates = preprocessor.transform(df_sub)
        postprocessor.learn(DataFrame(intermediates, columns=postprocessor.in_feature_names))
        self.out_feature_names = postprocessor.out_feature_names

    def apply(self, input_generator):
        return self._postprocessor.apply(
            self._preprocessor.extract_features(list(input_generator)))


def suggest_handlers(df, sample_size=10000, trees_optimized=True, hinted_featurizer=None):
    """
    Suggest handlers given the data
    :param df: DataFrame
    :param trees_optimized: if true, the handlers are suggested to optimize trees learning
     Otherwise, they are suggested for other learners (e.g. linear or NN)
    """
    import IPython

    assert isinstance(df, DataFrame)
    if len(df) > sample_size:
        df = df.ix[random.sample(list(df.index), sample_size)]

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

        dtype = Counter(type(value) for value in df[col] if (value == value)).most_common(1)[0][0]
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
            samples = df_sub.ix[random.sample(list(df_sub.index), 5)]
            desc['Sample values'] = Series(
                [', '.join(('{:.2g}' if isinstance(val, float) else '{:}')
                           .format(val) for val in samples[col])
                 for col in samples], index=cols)
            desc.columns = [col.title() for col in desc.columns]
            ret[kind] = desc
            IPython.display.display(desc)
            print

    return ret
