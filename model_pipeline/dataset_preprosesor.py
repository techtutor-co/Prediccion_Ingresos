import json
from sklearn.base import BaseEstimator, TransformerMixin
from ml_utils.data import DataUtils

class DatasetPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, config_path):
        with open(config_path) as json_file:
            self._config_data = json.load(json_file)
            self._data_utils = DataUtils(self._config_data)

    def fit(self, X, y = None):
        return None

    def transform(self, X, y = None):
        categories = self._data_utils.get_cols_by_type(X, 'object')
        return self._data_utils.encode_categories(X, categories)