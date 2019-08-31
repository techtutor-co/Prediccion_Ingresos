from ml_utils.data import DataUtils
from ml_utils.model import ModelUtils
import json
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
import warnings

warnings.filterwarnings('ignore')
 
def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as json_file:
            config_data = json.load(json_file)
            data_utils = DataUtils(config_data)
            model_utils = ModelUtils(config_data)

            data_utils.config_dataset()
            census_df = pd.read_csv(config_data['dataset_path'])
            
            data_utils.encode_categories(census_df, data_utils.get_cols_by_type(census_df, 'object'))
            features = census_df.drop(['income'], axis = 1)
            label = census_df[['income']]

            X_train, X_test, y_train, y_test = data_utils.split_dataset(features, label, config_data['random_state'], config_data['test_size'])

            learner = LogisticRegression()

            learner = model_utils.train_model(learner, X_train, y_train)
            score = model_utils.eval_model(learner, accuracy_score, X_test, y_test)

            print(f'score: {score}')

            model_utils.save_model(learner, config_data['model_path'])


if __name__ == "__main__":
    main()