"""
© 2022, ETH Zurich
"""


import os
from rescoss_logp_ml.utils import NUM_INNER_FOLDS, TEST_SET_FRACTION, SEED
from rescoss_logp_ml.two_d_models.chemprop_tools import scaffold_split
import chemprop
import pandas as pd
import random
import numpy as np
import argparse

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # needed for torch (in chemprop)


def chunk(xs, n, seed):
    "Randomly split xs in n roughly similarly-sized chunks"
    # from https://codereview.stackexchange.com/questions/4872/pythonic-split-list-into-n-random-chunks-of-roughly-equal-size (Acccessed 04.04.22)
    ys = list(xs)
    random.seed(seed)
    random.shuffle(ys)
    size = len(ys) // n
    leftovers = ys[size * n :]
    for c in range(n):
        if leftovers:
            extra = [leftovers.pop()]
        else:
            extra = []
        yield ys[c * size : (c + 1) * size] + extra


def split_dataset(csv, output_basepath, split_type):
    df = pd.read_csv(csv)
    df = df.reset_index(drop=True)
    outer_basepath = os.path.join(output_basepath, split_type)
    os.makedirs(outer_basepath, exist_ok=True)

    if split_type == "random":
        # divide the molecules randomly into 1/TEST_SET_FRACTION parts, then choose the first one as test set
        assignments = list(chunk(df.index.tolist(), int(1 / TEST_SET_FRACTION), SEED))
        for split_no, assignment in enumerate(assignments):
            if split_no == 0:
                df.loc[assignment, "split"] = "test"
            else:
                df.loc[assignment, "split"] = "train_val"
        train_val_idx = df[df.split == "train_val"].index.to_numpy()
        df[df.split == "test"].to_csv(os.path.join(outer_basepath, "test.csv"))
        df[df.split == "train_val"].to_csv(os.path.join(outer_basepath, "train_val.csv"))

        # inner cross-validation loop

        assignments_inner = list(chunk(train_val_idx, NUM_INNER_FOLDS, SEED))
        for i in range(NUM_INNER_FOLDS):
            inner_basepath = os.path.join(outer_basepath, f"inner_fold_{i}")
            os.makedirs(inner_basepath, exist_ok=True)
            val_idx = np.asarray(assignments_inner[i])
            train_idx = [assignments_inner[x] for x in range(NUM_INNER_FOLDS) if x != i]
            train_idx = np.asarray([x for t in train_idx for x in t])
            df.loc[val_idx].to_csv(os.path.join(inner_basepath, "val.csv"))
            df.loc[train_idx].to_csv(os.path.join(inner_basepath, "train.csv"))

    elif split_type == "scaffold":
        scaffold_stats_file = os.path.join(outer_basepath, "scaffold_stats.txt")
        arguments = [
            "--data_path",
            csv,
            "--dataset_type",
            "regression",
            "--target_columns",
            "logp_rescoss",
            "logp_exp",
            "--smiles_columns",
            "smiles",
            "--seed",
            SEED,
            "--pytorch_seed",
            SEED,
        ]
        arguments = [str(a) for a in arguments]
        args = chemprop.args.TrainArgs().parse_args(arguments)
        args.task_names = chemprop.data.utils.get_task_names(
            path=args.data_path,
            smiles_columns=args.smiles_columns,
            target_columns=args.target_columns,
            ignore_columns=args.ignore_columns,
        )

        data = chemprop.data.utils.get_data(path=csv, args=args)

        train_val_fraction = 1 - TEST_SET_FRACTION
        data_sets, idxs, counts, mols_per_scaffold = scaffold_split(
            data, sizes=(train_val_fraction, TEST_SET_FRACTION), seed=SEED
        )

        train_val_set, test_set = data_sets
        train_val_idx, test_idx = idxs
        assert len(train_val_set) + len(test_set) == len(data)
        with open(scaffold_stats_file, "w") as f:
            f.write(
                f"{'Data subset':<20}{'Number of molecules':<30}{'Number of scaffolds':<30}{'Mean number of molecules per scaffold':<30}\n"
            )
            f.write(
                f"{'train_val_set':<20}{len(train_val_set):<30}{counts[0]:<30}{np.mean(mols_per_scaffold[0]):<30.2f}\n"
            )
            f.write(f"{'test_set':<20}{len(test_set):<30}{counts[1]:<30}{np.mean(mols_per_scaffold[1]):<30.2f}\n")
        df.loc[sorted(test_idx)].to_csv(
            os.path.join(outer_basepath, "test.csv")
        )  # sorted: use original df order rather than scaffold-based order
        df.loc[sorted(train_val_idx)].to_csv(os.path.join(outer_basepath, "train_val.csv"))

        # inner cross-validation loop
        data_sets, idxs, counts, mols_per_scaffold = scaffold_split(
            train_val_set, sizes=tuple([(1 / NUM_INNER_FOLDS)] * NUM_INNER_FOLDS), seed=SEED
        )

        for i in range(NUM_INNER_FOLDS):
            val_idx = np.asarray(idxs[i])
            train_idx = [idxs[x] for x in range(NUM_INNER_FOLDS) if x != i]
            train_idx = np.asarray([x for t in train_idx for x in t])  # flatten
            assert train_idx.shape[0] + val_idx.shape[0] == len(train_val_set)
            # train_idx & val_idx are relative to train_val_set, so need to map back to original idx in the df:
            train_set_idx_in_original_df = np.asarray(train_val_idx)[train_idx]
            val_set_idx_in_original_df = np.asarray(train_val_idx)[val_idx]

            with open(scaffold_stats_file, "a") as f:
                f.write(
                    f"{f'inner_fold_{i}':<20}{len(idxs[i]):<30}{counts[i]:<30}{np.mean(mols_per_scaffold[i]):<30.2f}\n"
                )

            inner_basepath = os.path.join(outer_basepath, f"inner_fold_{i}")
            os.makedirs(inner_basepath, exist_ok=True)

            df.loc[sorted(val_set_idx_in_original_df)].to_csv(os.path.join(inner_basepath, "val.csv"))
            df.loc[sorted(train_set_idx_in_original_df)].to_csv(os.path.join(inner_basepath, "train.csv"))
    elif split_type == "time":
        # divide the molecules into sets based on their registration date
        df["registration_date"] = df["registration_date"].astype("datetime64")
        df = df.sort_values(by="registration_date")
        num_test_mols = int(TEST_SET_FRACTION * len(df))
        num_train_val_mols = len(df) - num_test_mols
        df.iloc[-num_test_mols:].to_csv(
            os.path.join(outer_basepath, "test.csv")
        )  # take newest x molecules as test set
        df.iloc[:num_train_val_mols].to_csv(os.path.join(outer_basepath, "train_val.csv"))

        # inner cross-validation loop
        for i in range(NUM_INNER_FOLDS):
            # https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4, "Cross Validation on Time Series" (Accessed 22.04.22)
            inner_basepath = os.path.join(outer_basepath, f"inner_fold_{i}")
            os.makedirs(inner_basepath, exist_ok=True)
            # divide by NUM_INNER_FOLDS + 1 since we need 1x train as well, see imagine on website above
            val_start_idx = (i + 1) * int(num_train_val_mols / (NUM_INNER_FOLDS + 1))
            val_end_idx = (i + 2) * int(num_train_val_mols / (NUM_INNER_FOLDS + 1))
            val_end_idx = min([val_end_idx, num_train_val_mols])
            df.iloc[:val_start_idx].to_csv(os.path.join(inner_basepath, "train.csv"))
            df.iloc[val_start_idx:val_end_idx].to_csv(os.path.join(inner_basepath, "val.csv"))
    else:
        raise ValueError(f"split_type {split_type} not supported.")
    print(f"Saved outputs in {outer_basepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    parser.add_argument("--output_basepath", type=str)
    parser.add_argument("--split_type", type=str)
    args = parser.parse_args()

    split_dataset(args.csv, args.output_basepath, args.split_type)
