import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataUtils:

    def __init__(self, config):
        self._origin_path = config['origin_path']
        self._target_path = config['dataset_path']
        self._column_names = config['column_names']

    def config_dataset(self):
        census_df = pd.read_csv(self._origin_path, names = self._column_names)
        census_df.to_csv(self._target_path, index = False)

    def get_cols_by_type(self, df, type_name):
        types_df = df.dtypes.to_frame(name = 'dtypes')
        categorical = types_df[types_df['dtypes'] == type_name]
        return list(categorical.index)
    
    def encode_category(self, df, category):
        lbl_encoder = LabelEncoder()
        return lbl_encoder.fit_transform(df[category])
    
    def encode_categories(self, df, categories):
        for category in categories:
            df[category] = self.encode_category(df, category)

    def split_dataset(self, features, label, random_state, test_size):        
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test