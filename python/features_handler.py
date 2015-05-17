from abc import ABCMeta, abstractmethod
from bisect import bisect_left


class FeaturesHandler(object):
    __metaclass__ = ABCMeta

    def __init__(self, in_feature_names):
        self.in_feature_names = in_feature_names
        self.out_feature_names = None

    def size(self):
        return len(self.out_feature_names)

    @abstractmethod
    def apply(self, values):
        """
        :param values: sequence of feature values corresponding in_feature_names
        :return sequence of (index,value)
        """

    def train(self, df):
        """
        Train the handler from data
        :param df: data frame
        """
        pass


class NoHandler(FeaturesHandler):
    def __init__(self, in_feature_names):
        super(NoHandler, self).__init__(in_feature_names)
        self.out_feature_names = in_feature_names

    def apply(self, values):
        return enumerate(values)


# Mapper handler maps input features to output features via a function (e.g. a lambda)
class MapperHandler(FeaturesHandler):
    def __init__(self, in_feature_names, out_feature_names, mapper):
        super(self.__class__, self).__init__(in_feature_names)
        self.out_feature_names = out_feature_names
        self._mapper = mapper

    def apply(self, values):
        return enumerate(self._mapper(list(values)))


class BaseCategoricalHandler(FeaturesHandler):
    __metaclass__ = ABCMeta

    def apply(self, values):
        preprocessor = self._preprocessor
        return ((self._indices[i].get(preprocessor(value) if preprocessor else value), 1)
                for i, value in enumerate(values))


class CategoricalHandler(BaseCategoricalHandler):
    def __init__(self, in_feature_names, preprocessor=None):
        super(self.__class__, self).__init__(in_feature_names)
        self._preprocessor = preprocessor
        self._indices = []
        self.out_feature_names = []
        self._size = 0

    def train(self, df):
        preprocessor = self._preprocessor
        names = self.out_feature_names
        for key in self.in_feature_names:
            index = {}
            for value in df[key]:
                value = preprocessor(value)
                if value not in index:
                    index[value] = self._size
                    self._size += 1
                    names.append(key + '=' + str(value))
            self._indices.append(index)


class UntrainedCategoricalHandler(BaseCategoricalHandler):
    def __init__(self, features_info, preprocessor=None):
        self.in_feature_names = []
        self._indices = []
        names = []
        size = 0
        for key in features_info:
            self.in_feature_names.append(key)
            index = {}
            for category in features_info[key]:
                index[category] = size
                names.append(key + '=' + str(category))
                size += 1
            self._indices.append(index)

        self._size = size
        self.out_feature_names = names
        self._preprocessor = preprocessor


class BinNormalizer(FeaturesHandler):
    # ToDo: automatic binning given number of bins
    def __init__(self, in_feature_names, thresholds_groups):
        super(self.__class__, self).__init__(in_feature_names)
        self.out_feature_names = [name + '_binned' for name in in_feature_names]
        self._threshold_groups = thresholds_groups

    def apply(self, values):
        return ((i, bisect_left(self._threshold_groups[i], value))
                for i, value in enumerate(values))
