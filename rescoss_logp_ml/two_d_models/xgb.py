"""
© 2022, ETH Zurich
"""


import os

import pandas as pd
import xgboost as xgb
from rescoss_logp_ml.data_handling.data_handling import load_dataset
from rescoss_logp_ml.utils import SEED, get_output_dir
from sklearn.metrics import mean_absolute_error, mean_squared_error


class XGBModel(xgb.XGBRegressor):
    def __init__(self, random_state=SEED, **h_params):
        super().__init__()
        self.random_state = random_state
        self.max_depth = h_params["max_depth"]
        self.learning_rate = h_params["learning_rate"]
        self.n_estimators = h_params["n_estimators"]
        self.h_param = h_params

    def train(self, dataset_name, target, train_csv, test_csv, val_at_epoch=None, return_preds_savepath=False):
        # val_at_epoch ignored for non-NN models
        _, features_dict = load_dataset(dataset_name, "2d_features")
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        X_train = [features_dict[key] for key in df_train.identifier]
        X_test = [features_dict[key] for key in df_test.identifier]
        y_train = df_train[target].to_numpy()
        y_test = df_test[target].to_numpy()

        self.fit(X_train, y_train)
        test_preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, test_preds)
        rmse = mean_squared_error(y_test, test_preds, squared=False)

        if return_preds_savepath:
            output_dir = get_output_dir("xgb", train_csv, target, self.h_param)
            model_savepath = os.path.join(output_dir, "xgb.json")
            os.makedirs(os.path.dirname(model_savepath), exist_ok=True)
            self.save_model(model_savepath)
            return mae, rmse, None, test_preds, model_savepath
        else:
            return mae, rmse, None

