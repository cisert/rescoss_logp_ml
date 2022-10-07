"""
© 2022, ETH Zurich
"""


import os

from xgboost import XGBRegressor
import chemprop
import tempfile
import pandas as pd
import joblib
from rescoss_logp_ml.utils import ROOT_PATH
from rescoss_logp_ml.descriptors import get_2d_features


def get_model_savepath(model_name, split_type, target):
    if model_name == "rf":
        model_savepath = os.path.join(ROOT_PATH, "saved_models", "az_set", split_type, target, "rf.joblib")
    elif model_name == "lasso":
        model_savepath = os.path.join(ROOT_PATH, "saved_models", "az_set", split_type, target, "lasso.joblib")
    elif model_name == "xgb":
        model_savepath = os.path.join(ROOT_PATH, "saved_models", "az_set", split_type, target, "xgb.json")
    elif model_name.startswith("chemprop"):
        model_savepath = os.path.join(ROOT_PATH, "saved_models", "az_set", split_type, target, "chemprop.pt")
    return model_savepath


def predict_with_trained_model(smiles_list, model_name, split_type=None, target=None, model_savepath=None):
    # load trained model
    if model_savepath is None:
        model_savepath = get_model_savepath(model_name, split_type, target)

    # make predictions
    if model_name == "rf" or model_name == "lasso":
        model = joblib.load(model_savepath)
        X = get_2d_features(smiles_list)
        preds = model.predict(X)

    elif model_name == "xgb":
        model = XGBRegressor()
        model.load_model(model_savepath)
        X = get_2d_features(smiles_list)
        preds = model.predict(X)

    elif model_name == "chemprop":
        # put SMILES in a csv
        tmpdir = tempfile.mkdtemp()
        df = pd.DataFrame({"smiles": smiles_list})
        test_path = os.path.join(tmpdir, "test.csv")
        preds_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(test_path, index=False)

        # run the model
        arguments = ["--test_path", test_path, "--checkpoint_path", model_savepath, "--preds_path", preds_path]
        arguments = [str(a) for a in arguments]
        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args)

        # load SMILES from csv
        preds = pd.read_csv(preds_path)[target].tolist()
    else:
        raise ValueError("Invalid model name")
    return preds

