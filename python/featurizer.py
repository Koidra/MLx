import csv
import random
import numpy as np
from collections import Counter
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
        self.in_feature_types = None
        self.out_feature_names = None
        self._handlers_feature_indices = None

    # Contracts: df must contain only feature columns
    def learn(self, df, sample_size=None):
        self.in_feature_names = list(df.columns)
        self.in_feature_types = []
        for col in df:
            dtype = Counter(type(value)
                            for value in df[col] if (value == value)).most_common(1)[0][0]
            self.in_feature_types.append(np.float32 if dtype == np.float64
                                         else np.int32 if dtype == np.int64 else dtype)
        self.out_feature_names = []
        self._handlers_feature_indices = []  # indices of the input features for each handler

        if sample_size and len(df) > sample_size:
            df = df.ix[random.sample(df.index, sample_size)]

        name_to_index = {name: i for i, name in enumerate(df.columns)}
        for handler in self.handlers:
            handler.learn(df)
            self._handlers_feature_indices.append([name_to_index[name]
                                                   for name in handler.in_feature_names])
            self.out_feature_names.extend(handler.out_feature_names)

    def size(self):
        return len(self.out_feature_names)

    def get_active_features(self):
        features_set = set()
        for handler in self.handlers:
            features_set |= set(handler.in_feature_names)
        return sorted(features_set)

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

    def _transform(self, row_generator, row_processor, sparse):
        if sparse:
            ind_ptr = [0]
            indices = []
            data = []
            for row in row_generator:
                cols, values = self.get_features_sparse(row_processor(row))
                ind_ptr.append(len(indices))
                indices.extend(cols)
                data.extend(values)
            ind_ptr.append(len(indices))
            return csr_matrix((data, indices, ind_ptr),
                              shape=(len(ind_ptr)-1, self.size()),
                              dtype=np.float32)
        else:
            return np.array([self.get_features_dense(row_processor(row)) for row in row_generator],
                            dtype=np.float32)

    # Note: df cannot contain the label column
    def transform(self, raw_data, sparse=True):
        if isinstance(raw_data, DataFrame):
            return self._transform(raw_data.values, lambda x: x, sparse)

        assert isinstance(raw_data, str)  # filename

        def row_processor(row):
            values = [0] * dim
            for i, idx in enumerate(indexer):
                raw = row[idx]
                values[i] = 0 if (not raw and np.issubdtype(types[i], np.number)) \
                    else types[i](raw)
            return values

        types = self.in_feature_types
        dim = len(types)
        with open(raw_data, 'rb') as csv_file:
            reader = csv.reader(csv_file)
            header = reader.next()
            indexer = [header.index(f) for f in self.in_feature_names]
            return self._transform(reader, row_processor)