import numpy
import types
from abc import ABCMeta, abstractmethod
from bisect import bisect_left
from collections import Counter
from pandas import DataFrame
from .core import check


class FeaturesHandler(object):
    __metaclass__ = ABCMeta

    def __init__(self, in_feature_names):
        self.in_feature_names = in_feature_names
        self.out_feature_names = None

    def size(self):
        return len(self.out_feature_names)

    @abstractmethod
    def apply(self, input_generator):
        """
        :param input_generator: sequence of feature values corresponding to in_feature_names
        :return sequence of (index,value)
        """
        return enumerate(input_generator)

    def learn(self, df):
        """
        Train the handler from data
        :param df: data frame
        """
        assert isinstance(df, DataFrame)


class NoHandler(FeaturesHandler):
    def __init__(self, in_feature_names):
        super(NoHandler, self).__init__(in_feature_names)
        self.out_feature_names = in_feature_names

    def apply(self, input_generator):
        return enumerate(input_generator)


class BoolHandler(NoHandler):
    """
    This is nothing but a NoHandler with a different name
    """


# Mapper handler maps input features to output features via a function (e.g. a lambda)
class MapperHandler(FeaturesHandler):
    def __init__(self, in_feature_names, out_feature_names, mapper):
        """
        :param in_feature_names: list of input feature names
        :param out_feature_names: list of output feature names
        :param mapper: a function that takes a list and return another list
        """
        super(self.__class__, self).__init__(in_feature_names)
        self.out_feature_names = out_feature_names
        self._mapper = mapper

    def apply(self, input_generator):
        return enumerate(self._mapper(list(input_generator)))


class OneToOneMapperHandler(FeaturesHandler):
    def __init__(self, in_feature_name, out_feature_name, mapper):
        self.in_feature_names = [in_feature_name]
        self.out_feature_names = [out_feature_name]
        self._mapper = mapper

    def apply(self, input_generator):
        yield 0, self._mapper(input_generator.next())


class CategoricalHandler(FeaturesHandler):
    def __init__(self, in_feature_names, cats={}, preprocessor=None):
        for in_feature_name in cats:
            if in_feature_name not in in_feature_names:
                raise ValueError('cats contains keys ({0}) not in feature names'.format(in_feature_name))
        super(self.__class__, self).__init__(in_feature_names)

        maps = [None] * len(in_feature_names)
        count = 0  # total number of unique values
        out_feature_names = []
        for in_feature_name in cats:
            _map = {}
            for value in cats[in_feature_name]:
                _map[value] = count
                out_feature_names.append(in_feature_name + '=' + str(value))
                count += 1
            maps[in_feature_names.index(in_feature_name)] = _map

        self.out_feature_names = out_feature_names
        self._maps = maps
        self._count = count
        self._preprocessor = lambda x: x if preprocessor is None else preprocessor

    def learn(self, df):
        maps = self._maps
        preprocessor = self._preprocessor
        out_feature_names = self.out_feature_names
        for i, in_feature_name in enumerate(self.in_feature_names):
            if maps[i] is not None:
                continue
            _map = {}
            for value in df[in_feature_name]:
                value = preprocessor(value)
                if value not in _map:
                    _map[value] = self._count
                    out_feature_names.append(in_feature_name + '=' + str(value))
                    self._count += 1
            maps[i] = _map

    def apply(self, input_generator):
        preprocessor = self._preprocessor
        for i, value in enumerate(input_generator):
            index = self._maps[i].get(preprocessor(value))
            if index is not None:
                yield index, 1


class BinNormalizer(FeaturesHandler):
    def __init__(self, in_feature_names, n_bins=32, thresholds_groups=None):
        """
        :param in_feature_names: names of input features
        :param n_bins: number of bins for each input feature
            A small number of bins (10-20) is typically sufficient for tree learners.
            For other learners (e.g. linear or NN), use a larger number of bins (100-1000).
        :param thresholds_groups: thresholds for each group
            If provided, n_bins will be ignored.
        """
        super(self.__class__, self).__init__(in_feature_names)
        self.out_feature_names = [name + '_bin' for name in in_feature_names]
        check(thresholds_groups is None
              or (len(thresholds_groups) == len(in_feature_names) and n_bins > 1))
        self._threshold_groups = thresholds_groups
        self._n_bins = n_bins

    def learn(self, df):
        if self._threshold_groups is not None:
            return

        self._threshold_groups = []
        for col in self.in_feature_names:
            f_values = df[col].dropna()
            assert numpy.issubdtype(f_values, numpy.number)
            histogram = sorted(Counter(f_values).items())
            if len(histogram) <= self._n_bins:
                self._threshold_groups.append(None)
                continue

            thresholds = []
            n_bins = self._n_bins
            n_items = float(len(f_values))
            ih = 0
            ih_lim = len(histogram)
            while n_bins > 1:
                expected_bin_size = n_items / n_bins
                count = 0
                while count < expected_bin_size:
                    count += histogram[ih][1]
                    ih += 1
                last_addition = histogram[ih-1][1]
                if count - expected_bin_size > expected_bin_size - (count - last_addition) \
                        and count != last_addition:
                    ih -= 1
                    count -= last_addition
                if ih == ih_lim:  # reaching the end before fulfilling the number of bins
                    break
                thresholds.append((histogram[ih-1][0] + histogram[ih][0])/2)
                n_items -= count
                n_bins -= 1
            self._threshold_groups.append(thresholds)

    def apply(self, input_generator):
        for i, value in enumerate(input_generator):
            thresholds = self._threshold_groups[i]
            yield i, (bisect_left(thresholds, value) if thresholds else value)


class MinMaxNormalizer(FeaturesHandler):
    def __init__(self, in_feature_names):
        super(MinMaxNormalizer, self).__init__(in_feature_names)
        raise RuntimeError('Not yet implemented')

    def learn(self, df):
        pass

    def apply(self, input_generator):
        pass


class PredicatesHandler(FeaturesHandler):
    """
    The predicates handler computes predicate features
    Those predicates are defined as normal functions in a module
    For each predicate:
        - The function name is treated as output feature name
        - The function argument variables are treated as input feature names
    """

    def __init__(self, module):
        predicates = []
        features_set = set()
        out_feature_names = []
        for f in dir(module):
            if (not isinstance(module.__dict__.get(f), types.FunctionType)) or f.startswith('_'):
                continue  # filter out elements that are not public functions
            predicate = module.__dict__.get(f)
            predicates.append(predicate)
            out_feature_names.append(f)
            features_set |= set(predicate.func_code.co_varnames)
        in_feature_names = list(features_set)
        name_to_idx = {name: idx for idx, name in enumerate(in_feature_names)}
        self._indices = [[name_to_idx[name] for name in predicate.func_code.co_varnames]
                         for predicate in predicates]
        self._predicates = predicates
        self.in_feature_names = in_feature_names
        self.out_feature_names = out_feature_names

    def apply(self, input_generator):
        values = list(input_generator)
        indices = self._indices
        return enumerate(predicate(*[values[j] for j in indices[i]])
                         for i, predicate in enumerate(self._predicates))
