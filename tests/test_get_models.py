"""
© 2022, ETH Zurich
"""


from rescoss_logp_ml.utils import HPARAMS
from rescoss_logp_ml.run_model import get_model, get_h_param_combinations
import pytest


@pytest.mark.parametrize("model_name", ["rf", "lasso", "xgb", "chemprop"])
def test_getting_models(model_name):
    h_params = get_h_param_combinations(HPARAMS[model_name])[0]
    model = get_model(model_name, **h_params)


if __name__ == "__main__":
    test_getting_models("rf")
