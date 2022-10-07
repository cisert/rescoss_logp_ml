"""
© 2022, ETH Zurich
"""


from rescoss_logp_ml.utils import (
    FOLDER_NAMES,
    PROC_DATA_PATH,
    NUM_INNER_FOLDS,
    HPARAMS,
    LEARNING_CURVE_SIZES,
    get_output_dir,
)
import os
import numpy as np
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
import argparse
import glob
from rescoss_logp_ml.data_handling import statsig
from rescoss_logp_ml.run_model import run_single_model


def collect_results_for_dataset(dataset_name, target, models):
    split_types = ["random", "scaffold", "time"]
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([split_types, ["mae", "rmse"], ["value", "lower_error", "upper_error"]])
    )
    df.loc[:, "model"] = models
    df = df.set_index("model")
    for split_type in split_types:
        for model_name in models:
            all_res = get_final_results(dataset_name, target, split_type, model_name)
            if all_res is not None:
                (
                    mae,
                    rmse,
                    num_best_epochs,
                    preds,
                    model_savepath,
                    df_hyperparameter_results,
                    test_csv,
                    statsig_res,
                ) = all_res
                mae_stat, mae_stat_le, mae_stat_ue, rmse_stat, rmse_stat_le, rmse_stat_up = statsig_res
                df.loc[model_name, (split_type, "mae")] = mae_stat, mae_stat_le, mae_stat_ue
                df.loc[model_name, (split_type, "rmse")] = rmse_stat, rmse_stat_le, rmse_stat_up
            else:
                continue
    return df


def format_hyperparameter_results(errors_inner, h_param_combinations):
    res = []
    for i, params in enumerate(h_param_combinations):
        maes = [e["mae"][i] for e in errors_inner]
        rmses = [e["rmse"][i] for e in errors_inner]
        this_res = dict(
            params,
            **{
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "rmse_mean": np.mean(rmses),
                "rmse_std": np.std(rmses),
            },
        )
        res.append(this_res)
    df_hyperparameter_results = pd.DataFrame(res)
    return df_hyperparameter_results


def get_h_param_combinations(h_params):
    """Get all combination from the different hyperparameters.

    Args:
        h_params (dict): dictionary with different hyperparameters and their ranges

    Returns:
        list(dict): hyperparameter combinations
    """
    combinations = list(itertools.product(*h_params.values()))
    return [{key: val for key, val in zip(h_params.keys(), vals)} for vals in combinations]


def get_final_results(dataset_name, target, split_type, model_name_for_hparams):
    h_param_combinations = get_h_param_combinations(HPARAMS[model_name_for_hparams])
    basepath = os.path.join(PROC_DATA_PATH, FOLDER_NAMES[dataset_name], "dataset_splits", split_type)
    test_csv = os.path.join(basepath, "test.csv")
    train_val_csv = os.path.join(basepath, "train_val.csv")
    completed_results = []

    # inner cross-validation loop
    errors_inner = {}
    for j in range(NUM_INNER_FOLDS):
        inner_basepath = os.path.join(basepath, f"inner_fold_{j}")
        train_csv = os.path.join(inner_basepath, "train.csv")
        val_csv = os.path.join(inner_basepath, "val.csv")

        # hyperparameter optimization loop
        errors_hparams = []
        for h_param in h_param_combinations:
            results_path = os.path.join(
                get_output_dir(model_name_for_hparams, train_csv, target, h_param), "results.pkl"
            )
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    mae, rmse, best_num_epochs = pickle.load(f)
                errors_hparams.append(dict({"mae": mae, "rmse": rmse, "best_num_epochs": best_num_epochs}, **h_param))
            else:
                continue
        errors_inner[f"fold_{j}"] = errors_hparams
    df_hyperparameter_results = None
    try:
        df_hyperparameter_results = collect_results(errors_inner)
        best_h_params, best_num_epochs = get_best_hyperparameters(df_hyperparameter_results)
        results_path = os.path.join(
            get_output_dir(model_name_for_hparams, train_val_csv, target, best_h_params), "results.pkl"
        )
    except:
        results_path = "not_existing"

    if not os.path.exists(results_path):
        results_paths = os.path.dirname(
            os.path.join(get_output_dir(model_name_for_hparams, train_val_csv, target, h_param_combinations[0]))
        )
        results_paths_res = glob.glob(os.path.join(results_paths, "*", "results.pkl"))
        if len(results_paths_res) > 1:
            raise ValueError
        elif len(results_paths_res) == 0:
            return None
        else:
            results_path = results_paths_res[0]
    print("Found final results")
    with open(results_path, "rb") as f:
        mae, rmse, num_best_epochs, preds, model_savepath = pickle.load(f)
    y = pd.read_csv(test_csv)[target].to_numpy()
    mae_stat, mae_stat_le, mae_stat_ue = statsig.mae(y, preds)
    rmse_stat, rmse_stat_le, rmse_stat_up = statsig.rmse(y, preds)
    statsig_res = mae_stat, mae_stat_le, mae_stat_ue, rmse_stat, rmse_stat_le, rmse_stat_up
    return mae, rmse, num_best_epochs, preds, model_savepath, df_hyperparameter_results, test_csv, statsig_res


def collect_results(errors_inner):
    dfs = []
    for fold, vals in errors_inner.items():
        df = pd.DataFrame(vals)
        df.loc[:, "fold"] = fold
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def get_best_hyperparameters(df):
    h_param_names = [x for x in list(df.keys()) if x not in ["mae", "rmse", "best_num_epochs", "fold"]]
    best_h_param_vals = df.groupby(by=h_param_names).mae.mean().idxmin()
    if len(h_param_names) > 1:
        best_h_params = {key: val for key, val in zip(h_param_names, best_h_param_vals)}
    else:
        best_h_params = {h_param_names[0]: best_h_param_vals}

    # average (over folds) number of epochs with this h_param combination
    if df.best_num_epochs.unique()[0] is None:
        best_num_epochs = None
    else:
        best_num_epochs = int(
            df.loc[(df[list(best_h_params)] == pd.Series(best_h_params)).all(axis=1)].best_num_epochs.mean()
        )
    return best_h_params, best_num_epochs


def run_model(
    dataset_name, model_name_for_hparams, target, basepath, no_overwrite, auto_submit_if_all_experiments_done
):
    results = None
    h_param_combinations = get_h_param_combinations(HPARAMS[model_name_for_hparams])

    test_csv = os.path.join(basepath, "test.csv")
    train_val_csv = os.path.join(basepath, "train_val.csv")
    completed_results = []

    # inner cross-validation loop
    errors_inner = {}
    for j in range(NUM_INNER_FOLDS):
        print(f"{j+1}/{NUM_INNER_FOLDS} inner loop", flush=True)
        inner_basepath = os.path.join(basepath, f"inner_fold_{j}")
        train_csv = os.path.join(inner_basepath, "train.csv")
        val_csv = os.path.join(inner_basepath, "val.csv")

        errors_hparams = []
        # hyperparameter optimization loop
        for h_param in h_param_combinations:
            results_path = os.path.join(
                get_output_dir(model_name_for_hparams, train_csv, target, h_param), "results.pkl"
            )
            if os.path.exists(results_path):
                # collect results
                completed_results.append(results_path)
                with open(results_path, "rb") as f:
                    mae, rmse, best_num_epochs = pickle.load(f)
                errors_hparams.append(dict({"mae": mae, "rmse": rmse, "best_num_epochs": best_num_epochs}, **h_param))
            else:
                # compute results
                run_single_model(
                    model_name=model_name_for_hparams,
                    dataset_name=dataset_name,
                    target=target,
                    train_csv=train_csv,
                    test_csv=val_csv,
                    val_at_epoch=None,  # validate at each epoch
                    return_preds_savepath=False,
                    **h_param,
                )
        errors_inner[f"fold_{j}"] = errors_hparams

    # format hyperparameter results
    if len(completed_results):
        df_hyperparameter_results = collect_results(errors_inner)

        # pick best hyperparameters: for each hyperparameter combination, take average over the NUM_INNER_FOLDS, then choose best one
        best_h_params, best_num_epochs = get_best_hyperparameters(df_hyperparameter_results)

        results_path = os.path.join(
            get_output_dir(model_name_for_hparams, train_val_csv, target, best_h_params), "results.pkl"
        )
        results_exist = os.path.exists(results_path)

        if results_exist:
            if no_overwrite:
                return
            else:
                if input(f"{results_path} exists. Recompute? [y/n]\n").lower() != "y":
                    # collect results
                    with open(results_path, "rb") as f:
                        mae, rmse, _, preds, model_savepath = pickle.load(f)
                        # make sure error was computed correctly
                        assert np.isclose(
                            mean_squared_error(pd.read_csv(test_csv)[target].to_numpy(), preds, squared=False), rmse,
                        )
                        assert np.isclose(np.mean(np.abs(pd.read_csv(test_csv)[target].to_numpy() - preds)), mae)
                        results = (
                            mae,
                            rmse,
                            best_num_epochs,
                            preds,
                            model_savepath,
                            test_csv,
                            train_val_csv,
                            df_hyperparameter_results,
                        )
        else:  # results don't exist --> need to compute them
            print(f"Found results from {len(df_hyperparameter_results)} experiments.")
            print(f"Best hyperparameters: {best_h_params}")
            if not auto_submit_if_all_experiments_done:  # ask what to do
                c = input("Continue? (y/n)")
                if c.lower() != "y":
                    return
            else:  # auto submit --> only stop in case we don't have all results yet
                num_completed_experiments = len(df_hyperparameter_results)
                num_required_experiments = len(h_param_combinations) * NUM_INNER_FOLDS
                if num_completed_experiments != num_required_experiments:
                    return

            # compute results
            run_single_model(
                model_name=model_name_for_hparams,
                dataset_name=dataset_name,
                target=target,
                train_csv=train_val_csv,
                test_csv=test_csv,
                val_at_epoch=best_num_epochs,
                return_preds_savepath=True,
                **best_h_params,
            )
            results = None
        # can't choose lowest test set error for any epoch but need to fix num epochs as a hyperparameter!
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--target", type=str, default="logp_rescoss")
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--learning_curve", action="store_true", default=False)
    parser.add_argument("--no_overwrite", action="store_true", default=False)
    parser.add_argument("--auto_submit_if_all_experiments_done", action="store_true", default=False)
    args = parser.parse_args()

    if args.learning_curve:
        ns = LEARNING_CURVE_SIZES[args.dataset]
        for n in ns:
            basepath = os.path.join(
                PROC_DATA_PATH, FOLDER_NAMES[args.dataset], "learning_curves", args.split_type, f"size_{n}",
            )
            run_model(
                args.dataset,
                args.model,
                args.target,
                basepath,
                args.no_overwrite,
                args.auto_submit_if_all_experiments_done,
            )
    else:
        basepath = os.path.join(PROC_DATA_PATH, FOLDER_NAMES[args.dataset], "dataset_splits", args.split_type)
        run_model(
            args.dataset,
            args.model,
            args.target,
            basepath,
            args.no_overwrite,
            args.auto_submit_if_all_experiments_done,
        )

