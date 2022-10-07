"""
© 2022, ETH Zurich
"""

from rescoss_logp_ml.utils import PROC_DATA_PATH, FOLDER_NAMES
import pickle
import os

def load_dataset(dataset, feature_type):
    savepath = os.path.join(PROC_DATA_PATH, FOLDER_NAMES[dataset], f"{dataset}_{feature_type}.pkl",)
    with open(savepath, "rb") as handle:
        df, features_dict = pickle.load(handle)
    return df, features_dict

