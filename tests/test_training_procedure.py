"""
© 2022, ETH Zurich
"""

import os
import numpy as np
from rescoss_logp_ml.run_training_for_model import get_final_results, run_model
from rescoss_logp_ml.utils import FOLDER_NAMES, PROC_DATA_PATH
import io


dataset = "az"
split_type = "random"
target = "logp_rescoss"
model_name = "rf_test_only"
basepath = os.path.join(PROC_DATA_PATH, FOLDER_NAMES[dataset], "dataset_splits", split_type)
no_overwrite = False
auto_submit_if_all_experiments_done = True


def test_training_procedure_cross_val(monkeypatch):
    # test cross-validation for hyperparameter selection
    monkeypatch.setattr("sys.stdin", io.StringIO("y"))  # confirm recomputation of test values
    run_model(
        dataset, model_name, target, basepath, no_overwrite, auto_submit_if_all_experiments_done,
    )


def test_training_procedure_final_model(monkeypatch):
    # need to run this again after cross-validation test to get final model
    monkeypatch.setattr("sys.stdin", io.StringIO("y"))  # confirm recomputation of test values
    run_model(
        dataset, model_name, target, basepath, no_overwrite, auto_submit_if_all_experiments_done,
    )
    mae = get_final_results(dataset, target, split_type, model_name)[0]
    expected_mae = 0.58398  # tested initially with limited hyperparameters (rf_test_only, for quick testing)
    assert np.isclose(mae, expected_mae)