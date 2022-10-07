"""
© 2022, ETH Zurich
"""


import json
import os

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
PROC_DATA_PATH = os.path.join(ROOT_PATH, "proc_data")

FOLDER_NAMES = {"az": "az_set", "inhouse": "inhouse_data"}
MODEL_FILENAMES = {"rf": "rf.joblib", "lasso": "lasso.joblib", "xgb": "xgb.json", "chemprop": "chemprop.pt"}
SEED = 1234
TEST_SET_FRACTION = 0.2
NUM_INNER_FOLDS = 5  # number of splits for cross-validation in inner loop
LOGP_RESCOSS_CUTOFF = 8  # predictions above this are typically not good

with open(os.path.join(ROOT_PATH, "hparams.json"), "r") as f:
    HPARAMS = json.load(f)
HPARAMS["rf_fp_only"] = HPARAMS["rf"]

with open(os.path.join(ROOT_PATH, "learning_curve_sizes.json"), "r") as f:
    LEARNING_CURVE_SIZES = json.load(f)


def get_output_dir(model_name, train_csv, target, h_param):
    output_dir = os.path.join(
        os.path.dirname(train_csv),
        f"results_{model_name}",
        target,
        "_".join([f"{key}{val}" for key, val in sorted(h_param.items())]),
    )
    return output_dir
