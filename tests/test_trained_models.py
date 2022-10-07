"""
© 2022, ETH Zurich
"""


from rescoss_logp_ml.predict_with_trained_model import predict_with_trained_model
from rescoss_logp_ml.utils import ROOT_PATH, DATA_PATH, MODEL_FILENAMES
import os
import pandas as pd
import numpy as np
import pytest
import itertools

TEST_FIRST_N = 10  # faster tests and just want to check if saved models are working properly

models = ["rf", "lasso", "xgb", "chemprop"]
split_types = ["random", "scaffold"]
targets = ["logp_rescoss", "logp_exp"]
test_cases = list(itertools.product(models, split_types, targets))


@pytest.mark.parametrize(("model_name", "split_type", "target"), tuple(test_cases))
def test_single_model(model_name, split_type, target):
    """Helper function to run predictions with a single model and compare to what we expect

    Args:
        model_name (str): name of model
    """

    df = pd.read_csv(os.path.join(DATA_PATH, f"az_{split_type}_split_predictions.csv"))
    preds = predict_with_trained_model(
        df.smiles.tolist()[:TEST_FIRST_N], model_name=model_name, split_type=split_type, target=target,
    )
    preds = np.asarray(preds)
    assert np.allclose(df[f"{target}_pred_{model_name}"].to_numpy()[:TEST_FIRST_N], preds)
