"""
© 2022, ETH Zurich
"""


import os
import joblib
import pandas as pd
from rescoss_logp_ml.data_handling.data_handling import load_dataset
from rescoss_logp_ml.utils import SEED, get_output_dir
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUM_DESCRIPTORS = 200


class RandomForestPipe(Pipeline):
    def __init__(self, random_state=SEED, test_only=False, **h_params):
        self.random_state = random_state
        self.h_param = h_params
        self.test_only = test_only
        scaler = StandardScaler()
        rf = RandomForestRegressor(random_state=random_state, criterion="squared_error", **h_params)
        super().__init__([("scaler", scaler), ("rf", rf)])

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
            if self.test_only:
                output_dir = get_output_dir("rf_test_only", train_csv, target, self.h_param)
            else:
                output_dir = get_output_dir("rf", train_csv, target, self.h_param)
            model_savepath = os.path.join(output_dir, "rf.joblib")
            os.makedirs(os.path.dirname(model_savepath), exist_ok=True)
            joblib.dump(self, model_savepath)
            return mae, rmse, None, test_preds, model_savepath
        else:
            return mae, rmse, None

