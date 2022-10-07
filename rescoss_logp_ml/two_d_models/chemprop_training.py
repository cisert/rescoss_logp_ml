"""
© 2022, ETH Zurich
"""


import os
import chemprop
from rescoss_logp_ml.two_d_models.chemprop_tools import run_training
from rescoss_logp_ml.utils import SEED, get_output_dir


class ChempropModel:
    def __init__(self, random_state=SEED, **h_params):
        self.random_state = random_state
        super().__init__()
        self.h_param = h_params
        self.depth = h_params["depth"]
        self.hidden_size = h_params["hidden_size"]

    def train(self, dataset_name, target, train_csv, test_csv, val_at_epoch=None, return_preds_savepath=False):
        # NOTE: since we're not using chemprops internal validation scheme, but need to pass a test & validation set,
        #       we simply use the test_csv twice and discard the test-set result from chemprop (using only the validation numbers)
        save_dir = os.path.join(get_output_dir("chemprop", train_csv, target, self.h_param), "chemprop_logs")
        arguments = [
            "--data_path",
            train_csv,
            "--separate_test_path",
            test_csv,
            "--separate_val_path",
            test_csv,
            "--dataset_type",
            "regression",
            "--save_dir",
            save_dir,
            "--target_columns",
            target,
            "--smiles_columns",
            "smiles",
            "--seed",
            SEED,
            "--pytorch_seed",
            SEED,
            "--epochs",  # max number
            100,
            "--quiet",
            "--num_folds",
            1,  # we're doing the folds ourselves in run_model.py
            "--loss_function",
            "mse",
            "--metric",
            "mae",  # used to choose best epoch number
            "--extra_metrics",
            "rmse",
            "--save_preds",
            "--depth",
            self.depth,
            "--hidden_size",
            self.hidden_size,
            "--num_workers",
            2,
        ]
        arguments = [str(a) for a in arguments]
        args = chemprop.args.TrainArgs().parse_args(arguments)
        logger = chemprop.utils.create_logger(name="logger", save_dir=args.save_dir, quiet=args.quiet)
        # Initialize relevant variables
        save_dir = args.save_dir
        args.task_names = chemprop.data.utils.get_task_names(
            path=args.data_path,
            smiles_columns=args.smiles_columns,
            target_columns=args.target_columns,
            ignore_columns=args.ignore_columns,
        )

        data = chemprop.data.utils.get_data(path=train_csv, args=args)

        val_scores, best_epoch, test_scores, test_preds, model_savepath = run_training(
            args=args, data=data, logger=logger, val_at_epoch=val_at_epoch
        )
        if return_preds_savepath:
            return val_scores["mae"], val_scores["rmse"], best_epoch, test_preds, model_savepath
        else:
            return val_scores["mae"], val_scores["rmse"], best_epoch

