import random
from numpy import array, float32
from pandas import DataFrame
from scipy.sparse import csr_matrix
from features_handler import FeaturesHandler

class Featurizer:
    def __init__(self, handlers):
        assert isinstance(handlers, list)
        for handler in handlers:
            assert isinstance(handler, FeaturesHandler)
        self.handlers = handlers
        self.in_feature_names = None
        self.out_feature_names = None
        self._handlers_feature_indices = None

    # Contracts: df must contain only feature columns
    def learn(self, df, sample_size=100000):
        self._handlers_feature_indices = []  # indices of the input features for each handler
        self.in_feature_names = []
        self.out_feature_names = []

        if len(df) > sample_size:
            df = df.ix[random.sample(df.index, sample_size)]

        self.in_feature_names = list(df.columns)
        name_to_index = {name: i for i, name in enumerate(df.columns)}
        for handler in self.handlers:
            handler.learn(df)
            self._handlers_feature_indices.append([name_to_index[name]
                                                   for name in handler.in_feature_names])
            self.out_feature_names.extend(handler.out_feature_names)

    # Note that we pass through NaNs
    # So either the handlers or the learner needs to handle NaNs
    # The featurizer itself doesn't handle NaNs
    def _build_features(self, record, builder):
        offset = 0
        for h, handler in enumerate(self.handlers):
            feature_indices = self._handlers_feature_indices[h]
            for index, value in handler.apply(record[i] for i in feature_indices):
                if value:
                    builder(offset + index, value)
            offset += handler.size()

    def get_features_dense(self, record):
        """
        :param record: list of input feature values
        :return: list of transformed features
        """
        values = [0] * self.size()

        def builder(i, v):
            values[i] = v

        self._build_features(record, builder)
        return values

    def get_features_sparse(self, record):
            indices = []
            values = []

            def builder(i, v):
                indices.append(i)
                values.append(v)

            self._build_features(record, builder)
            return indices, values

    # Note: df cannot contain the label column
    def transform(self, df, sparse=True, learn=False, learning_sample_size=100000):
        assert isinstance(df, DataFrame)
        if learn:
            self.learn(df, learning_sample_size)

        features = df.values
        if sparse:
            data = []
            row_indices = []
            col_indices = []
            for i, row in enumerate(features):
                cols, values = self.get_features_sparse(row)
                row_indices.extend([i] * len(cols))
                col_indices.extend(cols)
                data.extend(values)
            return csr_matrix((data, (row_indices, col_indices)),
                              shape=(len(df), self.size()), dtype=float32)
        else:
            return array([self.get_features_dense(row) for row in features], dtype=float32)

    def size(self):
        return len(self.out_feature_names)

    def get_active_features(self):
        features_set = set()
        for handler in self.handlers:
            features_set |= set(handler.in_feature_names)
        return sorted(features_set)