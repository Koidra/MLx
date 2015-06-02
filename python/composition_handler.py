from pandas import DataFrame
from features_handler import FeaturesHandler
from featurizer import Featurizer


class CompositionHandler(FeaturesHandler):
    def __init__(self, handler1, handler2):
        assert isinstance(handler2, FeaturesHandler)
        if isinstance(handler1, list):
            for handler in handler1:
                assert isinstance(handler, FeaturesHandler)
            self._preprocessor = Featurizer(handler1)
        else:
            assert isinstance(handler1, FeaturesHandler)
            self._preprocessor = Featurizer([handler1])
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
        intermediates = preprocessor.transform(df_sub, sparse=False)
        postprocessor.learn(DataFrame(intermediates, columns=postprocessor.in_feature_names))
        self.out_feature_names = postprocessor.out_feature_names

    def apply(self, input_generator):
        return self._postprocessor.apply(
            self._preprocessor.get_features_dense(list(input_generator)))
