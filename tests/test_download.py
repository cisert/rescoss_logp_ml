"""
© 2022, ETH Zurich
"""


import requests

from rescoss_logp_ml.data_handling.download import DATA_URL, SAVED_MODELS_URL


def test_islinkup():
    for remote in [DATA_URL, SAVED_MODELS_URL]:
        r = requests.head(remote)
        assert r.ok
