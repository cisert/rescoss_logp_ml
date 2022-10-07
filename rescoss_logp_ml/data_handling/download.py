"""
© 2022, ETH Zurich
"""


import os
import tarfile
from rescoss_logp_ml.utils import ROOT_PATH

import requests
from tqdm import tqdm

DATA_URL = (
    "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/571431/data.tar.gz?sequence=3&isAllowed=y"
)
SAVED_MODELS_URL = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/571466/saved_models.tar.gz?sequence=1&isAllowed=y"
# TODO: check links


def download(src, dest, extract_dest):
    r = requests.get(src, stream=True)
    tsize = int(r.headers.get("content-length", 0))
    progress = tqdm(total=tsize, unit="iB", unit_scale=True, position=0, leave=False)

    with open(dest, "wb") as handle:
        progress.set_description(os.path.basename(dest))
        for chunk in r.iter_content(chunk_size=1024):
            handle.write(chunk)
            progress.update(len(chunk))

    with tarfile.open(dest) as handle:
        handle.extractall(extract_dest)


if __name__ == "__main__":
    download(DATA_URL, os.path.join(ROOT_PATH, "data.tar.gz"), ROOT_PATH)
    download(SAVED_MODELS_URL, os.path.join(ROOT_PATH, "saved_models.tar.gz"), ROOT_PATH)

