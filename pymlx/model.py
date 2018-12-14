"""
MLx is a ML library being incubated. This file will be deprecated when the library is released.
"""
from pandas import DataFrame, Series
import dill
from abc import ABCMeta, abstractmethod
from .featurizer import Featurizer


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, predictor=None, featurizer=None):
        assert isinstance(featurizer, Featurizer)
        self.predictor = predictor
        self.featurizer = featurizer
        self._num_features = featurizer.size()

    @staticmethod
    def load(self, model_path):
        with open(model_path, "rb") as fp:
            featurizer = dill.load(fp)
            predictor = dill.load(fp)
            return Model(predictor, featurizer)

    def save(self, model_path):
        with open(model_path, "wb") as fp:
            dill.dump(self.featurizer, fp)
            dill.dump(self.predictor, fp)

    def get_feature_names(self):
        return self.featurizer.in_feature_names

    @abstractmethod
    def predict(self, test_data):
        pass

    def get_fscores(self, no_features_name=True):
        if self.predictor.__module__ == 'xgboost.sklearn':
            # for xgboost
            # feature_names = self.featurizer.out_feature_names
            _fscores = self.predictor.get_booster().get_fscore()
            fscores = {}
            if no_features_name: # when the training data is dataframe
                for k, v in _fscores.items():
                    fscores[self.featurizer.out_feature_names[int(k.replace('f', ''))]] = v
            else:
                fscores = _fscores

            # fscores = {feature_names[int(f[1:])]: fscores[f] for f in fscores}
            return Series(fscores).sort_values(ascending=False)
        else:
            # for lightgbm
            fscores = {}
            booster = self.predictor.booster_
            fimportances = booster.feature_importance()
            # print('light gbm', fimportances)
            for i, fname in enumerate(booster.feature_name()):
                if no_features_name:
                    fscores[self.featurizer.out_feature_names[int(fname.replace('Column_', ''))]] = fimportances[i]
                else:
                    fscores[fname] = fimportances[i]

            return Series(fscores).sort_values(ascending=False)


class BinaryClassifier(Model):
    def __init__(self, predictor=None, featurizer=None):
        super(BinaryClassifier, self).__init__(predictor, featurizer)

    def predict(self, test_data):
        if isinstance(test_data, DataFrame):
            test_data = self.featurizer.transform(test_data, return_dataframe=False)
        result = self.predictor.predict_proba(test_data)
        return result[:, 1]  # return probability of class 1


class Regressor(Model):
    def __init__(self, predictor=None, featurizer=None):
        super(Regressor, self).__init__(predictor, featurizer)

    def _is_distribution_predictor(self):
        return NotImplementedError('Check predictor is distribution predictor')

    def predict_mean(self, df):
        return NotImplementedError('Mean of y if predictor is dist predictor')
    
    def predict_mean_std(self, df):
        return NotImplementedError('Standard Deviation')
    
    def predict(self, test_data):
        if isinstance(test_data, DataFrame):
            test_data = self.featurizer.transform(test_data, return_dataframe=False)
        return self.predictor.predict(test_data)