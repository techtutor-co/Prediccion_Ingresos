import json
from sklearn.base import BaseEstimator, TransformerMixin
from ml_utils.model import ModelUtils

class ModelBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, config_path):
        with open(config_path) as json_file:
            self._config_data = json.load(json_file)
            self._model_utils = ModelUtils(self._config_data)
    
    def fit(self, X, y = None):
        return None
    
    def transform(self, X, y = None):
        return None