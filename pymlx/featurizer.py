import random
import numpy as np
import pandas as pd
from typing import Union, List
from scipy.sparse import csr_matrix
from .core import *
from .features_handler import *


def listable(t):
    return Union[t, List[t]]


class Featurizer:
    def __init__(self,
                 no_op: listable(str) = None,
                 categorical: listable(str) = None,
                 one_hot: listable(str) = None,
                 binning: listable(str) = None,
                 custom_handers: listable(FeaturesHandler) = None, sparse=False):

        _handlers = []

        if no_op:
            _handlers.append(NoHandler(no_op))

        if categorical:
            _handlers.append(IdEncodingHandler(categorical))

        if one_hot:
            _handlers.append(OneHotHandler(one_hot))

        if binning:
            _handlers.append(BinNormalizer(binning))

        if custom_handers:
            _handlers += custom_handers

        in_feature_names = set()

        for _handler in _handlers:
            in_feature_names |= set(_handler.in_feature_names)
        self.handlers = _handlers
        self.in_feature_names = sorted(in_feature_names)
        self.in_feature_types = None
        self.out_feature_names = None

        self._handlers_feature_indices = None
        self._initialize_features_vector = None
        self._sparse = sparse

        DTYPE = np.float32
        if sparse:
            self.to_matrix = lambda X: csr_matrix(X, shape=(len(X[2]) - 1, self.size()),
                                                  dtype=DTYPE)

            def add_feature(features, i, v):  # add one feature to a feature vector
                features[0].append(i)
                features[1].append(v)

            def add_features(data, features_vec):  # add a feature vector to a matrix
                data[0].extend(features_vec[1])  # data
                data[1].extend(features_vec[0])  # indices
                data[2].append(len(data[0]))  # ind_ptr

        else:
            self.to_matrix = lambda X: np.array(X, dtype=DTYPE)

            def add_feature(features, i, v):
                features[i] = v

            def add_features(data, features_vec):
                data.append(features_vec)

        self._add_feature = add_feature
        self._add_features = add_features

    def size(self):
        return len(self.out_feature_names)

    def learn(self, data: Union[str, DataFrame], sample_size: int=None):
        if isinstance(data, str):
            df = pd.read_csv(data, usecols=self.in_feature_names)
            # read_csv() doesn't honor the order of the input column names
            # So we need to re-order in_feature_names to make it consistent with the data order
            self.in_feature_names = list(df.columns)
        else:
            df = data[self.in_feature_names]

        if sample_size and len(df) > sample_size:
            df = df.ix[random.sample(list(df.index), sample_size)]

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

        self._initialize_features_vector = lambda: ([], []) if self._sparse else [0] * self.size()

    # Note that we pass through NaNs
    # So either the handlers or the learner needs to handle NaNs
    # The featurizer itself doesn't handle NaNs
    def featurize_row(self, row_raw: List):
        assert isinstance(row_raw, pd.Series)
        features = self._initialize_features_vector()
        add_feature = self._add_feature
        offset = 0
        for h, handler in enumerate(self.handlers):
            feature_indices = self._handlers_feature_indices[h]
            for index, value in handler.apply(row_raw[i] for i in feature_indices):
                if value:
                    add_feature(features, offset + index, value)
            offset += handler.size()
        return features

    # Note: df cannot contain the label column
    def transform(self, df_raw: DataFrame, return_dataframe=False):
        df = df_raw[self.in_feature_names]
        transformed = df.apply(self.featurize_row, axis=1)
        if not return_dataframe:
            return NotImplemented if self._sparse else np.array(transformed.values.tolist())
        else:
            # return features as a dataframe
            return NotImplemented if self._sparse else pd.DataFrame(data=transformed.values.tolist(), columns=self.out_feature_names)



class CompositionHandler(FeaturesHandler):
    def __init__(self, preprocessors: Union[FeaturesHandler, List[FeaturesHandler]], postprocessor: FeaturesHandler,
                 in_feature_names):
        super().__init__(in_feature_names)
        if isinstance(preprocessors, list):
            for handler in preprocessors:
                assert isinstance(handler, FeaturesHandler)
        else:
            assert isinstance(preprocessors, FeaturesHandler)
            preprocessors = [preprocessors]
        self._preprocessor = Featurizer(preprocessors, sparse=False)
        self._postprocessor = postprocessor
        self._intermediate_dimension = len(postprocessor.in_feature_names)
        self.in_feature_names = sorted(set().union(*[handler.in_feature_names
                                                     for handler in self._preprocessor.handlers]))

    def learn(self, df):
        preprocessor = self._preprocessor
        postprocessor = self._postprocessor
        df_sub = df[self.in_feature_names]
        preprocessor.learn(df_sub)
        intermediates = DataFrame(preprocessor.transform(df_sub), columns=preprocessor.out_feature_names)
        postprocessor.learn(intermediates)
        self.out_feature_names = postprocessor.out_feature_names

    def apply(self, input_generator):
        return self._postprocessor.apply(
            self._preprocessor.featurize_row(list(input_generator)))


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
    kinds = [OneHotHandler, 'TextHandler'] + numeric_kinds + \
            [CAT_INT, MapperHandler, CompositionHandler, PredicatesHandler, NA]

    handlers = {kind: [] for kind in kinds}
    priors = set()
    if hinted_featurizer:
        for handler in hinted_featurizer.handlers:
            for col in handler.in_feature_names:
                kind = handler.__class__
                if col not in handlers[kind]:
                    handlers[kind].append(col)
                    priors.add(col)
                    if kind in [OneHotHandler, BoolHandler] and df[col].dtype != object:
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
            kind = OneHotHandler
            # ToDo: could also be TEXT
        elif numpy.issubdtype(dtype, numpy.integer):
            col_low = col.lower()
            if col_low.endswith('id') or col_low.endswith('code') or col_low.endswith('type'):
                kind = OneHotHandler
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
    pd.options.display.float_format = '{:.2g}'.format
    pd.options.display.max_rows = 500
    for kind in kinds:
        if len(handlers[kind]) == 0:
            del handlers[kind]
        else:
            cols = sorted(handlers[kind])
            df_sub = df[cols]
            if kind == OneHotHandler or kind == BoolHandler:
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
            desc['Sample values'] = pd.Series(
                [', '.join(('{:.2g}' if isinstance(val, float) else '{:}')
                           .format(val) for val in samples[col])
                 for col in samples], index=cols)
            desc.columns = [col.title() for col in desc.columns]
            ret[kind] = desc
            IPython.display.display(desc)
            print

    return ret
