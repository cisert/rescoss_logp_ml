"""
© 2022, ETH Zurich
"""


import itertools
import os
import pickle

import numpy as np
import pandas as pd

from rescoss_logp_ml.utils import SEED, get_output_dir


def get_model(model_name, **h_params):
    """Construct specified model with hyperparameters. Imports done inside function for compatibility with different conda environments.

    Args:
        model_name (str): name of model, options: "rf", "lasso", "chemprop", "3d-mpnn"

    Returns:
        (type differs): model as specified, implements .train() method
    """
    if model_name.lower() == "rf":
        from rescoss_logp_ml.two_d_models.rf import RandomForestPipe

        model = RandomForestPipe(random_state=SEED, **h_params)

    elif model_name.lower() == "rf_test_only":
        from rescoss_logp_ml.two_d_models.rf import RandomForestPipe

        model = RandomForestPipe(random_state=SEED, test_only=True, **h_params)

    elif model_name.lower() == "lasso":
        from rescoss_logp_ml.two_d_models.lasso import LassoPipe

        model = LassoPipe(random_state=SEED, **h_params)

    elif model_name.lower() == "xgb":
        from rescoss_logp_ml.two_d_models.xgb import XGBModel

        model = XGBModel(random_state=SEED, **h_params)

    elif model_name.lower() == "chemprop":
        from rescoss_logp_ml.two_d_models.chemprop_training import ChempropModel

        model = ChempropModel(random_state=SEED, **h_params)

    else:
        raise ValueError("Unsupported model type")

    return model


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


def run_single_model(
    model_name, dataset_name, target, train_csv, test_csv, val_at_epoch, return_preds_savepath=False, **h_param
):
    model = get_model(model_name, **h_param)
    results = model.train(
        dataset_name=dataset_name,
        target=target,
        train_csv=train_csv,
        test_csv=test_csv,
        val_at_epoch=val_at_epoch,
        return_preds_savepath=return_preds_savepath,
    )
    if return_preds_savepath:
        mae, rmse, best_num_epochs, preds, model_savepath = results
    else:
        mae, rmse, best_num_epochs = results
    output_dir = get_output_dir(model_name, train_csv, target, h_param)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
