"""
© 2022, ETH Zurich
"""


import os
from rescoss_logp_ml.utils import NUM_INNER_FOLDS, LEARNING_CURVE_SIZES, SEED, PROC_DATA_PATH, FOLDER_NAMES
import pandas as pd
import argparse
import shutil


def get_sample(csv, n, split_type):
    df = pd.read_csv(csv)
    if split_type == "random" or split_type == "scaffold":
        # for scaffold, original split is already by scaffold, so we just randomly pick molecules for the learning curve training set from the original training set
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # randomly shuffle rows (seeded)
        df = df.iloc[:n]  # ensures nested training sets across the learning curve
    elif split_type == "time":
        df["registration_date"] = df["registration_date"].astype("datetime64")
        df = df.sort_values(by="registration_date")
        df = df.iloc[:n]
    else:
        raise ValueError("Unsupported split_type")
    return df


def split_dataset_for_learning_curve(dataset_name, split_type, n):
    outer_basepath = os.path.join(PROC_DATA_PATH, FOLDER_NAMES[dataset_name], "dataset_splits", split_type)
    lc_outer_basepath = os.path.join(
        PROC_DATA_PATH, FOLDER_NAMES[dataset_name], "learning_curves", split_type, f"size_{n}",
    )
    os.makedirs(os.path.dirname(lc_outer_basepath), exist_ok=True)
    os.makedirs(lc_outer_basepath, exist_ok=True)
    test_csv = os.path.join(outer_basepath, "test.csv")
    shutil.copy(test_csv, os.path.join(lc_outer_basepath, "test.csv"))
    train_val_csv = os.path.join(outer_basepath, "train_val.csv")
    df_train_val = get_sample(train_val_csv, n, split_type)
    df_train_val.to_csv(os.path.join(lc_outer_basepath, "train_val.csv"), index=False)
    for i in range(NUM_INNER_FOLDS):
        inner_basepath = os.path.join(outer_basepath, f"inner_fold_{i}")
        lc_inner_basepath = os.path.join(lc_outer_basepath, f"inner_fold_{i}")
        os.makedirs(lc_inner_basepath, exist_ok=True)
        train_csv = os.path.join(inner_basepath, "train.csv")
        val_csv = os.path.join(inner_basepath, "val.csv")
        shutil.copy(val_csv, os.path.join(lc_inner_basepath, "val.csv"))
        df_train = get_sample(train_csv, n, split_type)
        df_train.to_csv(os.path.join(lc_inner_basepath, "train.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--split_type", type=str)
    args = parser.parse_args()

    ns = LEARNING_CURVE_SIZES[args.dataset_name]
    for n in ns:
        split_dataset_for_learning_curve(args.dataset_name, args.split_type, n)
