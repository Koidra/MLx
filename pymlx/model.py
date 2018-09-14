"""
MLx is a ML library being incubated. This file will be deprecated when the library is released.
"""
from pandas import DataFrame, Series

import dill
from numpy import float32
from scipy.sparse import csr_matrix
from .featurizer import Featurizer


class BinaryClassifier:
    def __init__(self, predictor, featurizer):
        assert isinstance(featurizer, Featurizer)

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
    def predict(self, features):  # <-- meets error
        """
        :param features: list or array of feature values.
        Missing values MUST be encoded as nan, not None.
        :return: probability of the instance classified as True
        """

        indices, values = self.featurizer.featurize_row(features) # <-- input features must be a Pandas Series
        features = csr_matrix((values, indices, [0, len(values)]),
                              shape=(1, self._num_features),
                              dtype=float32)
        return self.predictor.predict_proba(features)[0][1]

    def bulk_predict(self, test_data):
        if isinstance(test_data, DataFrame):
            test_data = self.featurizer.transform(test_data, return_dataframe=True)
        # assert isinstance(test_data, csr_matrix) # <--
        result = self.predictor.predict_proba(test_data)
        return result[:, 1], result

    # REVIEW: this is specific to xgboost, which is temporary
    # ToDo:
    #   - Absorb xgboost
    #   - Each predictor should indicate whether it supports features importance
    #   11/09/2018: has just implement for xgboost and lightgbm
    def get_fscores(self):
        if self.predictor.__module__ == 'xgboost.sklearn':
            # for xgboost
            # feature_names = self.featurizer.out_feature_names
            fscores = self.predictor.get_booster().get_fscore()
            # fscores = {feature_names[int(f[1:])]: fscores[f] for f in fscores}
            return Series(fscores).sort_values(ascending=False)
        else:
            # for lightgbm
            fscores = {}
            booster = self.predictor.booster_
            fimportances = booster.feature_importance()
            for i, fname in enumerate(booster.feature_name()):
                fscores[fname] = fimportances[i]
            return Series(fscores).sort_values(ascending=False)


def load(filename):
    with open(filename, "rb") as fp:
        featurizer = dill.load(fp)
        predictor = dill.load(fp)
    return BinaryClassifier(predictor, featurizer)


