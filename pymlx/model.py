"""
MLx is a ML library being incubated. This file will be deprecated when the library is released.
"""
from pandas import DataFrame, Series

import dill
from numpy import float32
from scipy.sparse import csr_matrix
from xgboost import XGBModel
from .featurizer import Featurizer


class BinaryClassifier:
    def __init__(self, predictor, featurizer):
        assert isinstance(predictor, XGBModel) and isinstance(featurizer, Featurizer)
        self.predictor = predictor
        self.featurizer = featurizer
        self._num_features = featurizer.size()

    def save(self, filename):
        with open(filename, "wb") as fp:
            dill.dump(self.featurizer, fp)
            dill.dump(self.predictor, fp)

    def get_feature_names(self):
        """
        :return: list of features required by the model
        """
        return self.featurizer.in_feature_names

    # ToDo: implement scoring for row
    def predict(self, features):
        """
        :param features: list or array of feature values.
        Missing values MUST be encoded as nan, not None.
        :return: probability of the instance classified as True
        """
        indices, values = self.featurizer.extract_features(features)
        features = csr_matrix((values, indices, [0, len(values)]),
                              shape=(1, self._num_features),
                              dtype=float32)
        return self.predictor.predict_proba(features)[0][1]

    def bulk_predict(self, test_data):
        if isinstance(test_data, DataFrame):
            test_data = self.featurizer.transform(test_data)
        assert isinstance(test_data, csr_matrix)
        return self.predictor.predict_proba(test_data)[:, 1]

    # REVIEW: this is specific to xgboost, which is temporary
    # ToDo:
    #   - Absorb xgboost
    #   - Each predictor should indicate whether it supports features importance
    def get_fscores(self):
        feature_names = self.featurizer.out_feature_names
        fscores = self.predictor.booster().get_fscore()
        fscores = {feature_names[int(f[1:])]: fscores[f] for f in fscores}
        return Series(fscores).sort_values(ascending=False)


def load(filename):
    with open(filename, "rb") as fp:
        featurizer = dill.load(fp)
        predictor = dill.load(fp)
    return BinaryClassifier(predictor, featurizer)


