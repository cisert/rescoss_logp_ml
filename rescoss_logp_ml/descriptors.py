"""
© 2022, ETH Zurich
"""


from rdkit import Chem
from rdkit.Chem import DataStructs
import argparse
import os
from rescoss_logp_ml.utils import FOLDER_NAMES, PROC_DATA_PATH, DATA_PATH
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


def get_2d_features(smiles_list):
    from rdkit.Chem import rdMolDescriptors
    from descriptastorus.descriptors import rdNormalizedDescriptors

    generator = rdNormalizedDescriptors.RDKit2DNormalized()

    feature_list = []

    for smi in tqdm(smiles_list):
        results = generator.process(smi)
        processed, features = results[0], np.asarray(results[1:])
        if processed is False:
            print(f"Couldn't generate descriptors: {smi}")
            feature_list.append(np.array(float("NaN")))
        else:
            fp = rdMolDescriptors.GetHashedMorganFingerprint(Chem.MolFromSmiles(smi), radius=2, nBits=2048)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            combined = np.concatenate((features, fp_arr))
            feature_list.append(combined)
    return feature_list


def save_features(df, features_dict, data_set, feature_type):
    savepath = os.path.join(PROC_DATA_PATH, FOLDER_NAMES[args.data_set], f"{data_set}_{feature_type}_features.pkl",)
    with open(savepath, "wb") as handle:
        pickle.dump((df, features_dict), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {savepath}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_set", type=str, help="Which dataset to featurize")
    parser.add_argument("feature_type", type=str, help="Which feature (2d, 2d_graph, 3d_graph)")
    args = parser.parse_args()

    if args.data_set == "az":
        dataset_path = os.path.join(DATA_PATH, "az_final.csv")
    else:
        raise ValueError("Invalid dataset choice")
    df = pd.read_csv(dataset_path)

    if args.feature_type == "2d":
        features_list = get_2d_features(df.smiles)
        features_dict = {identifier: feature for identifier, feature in zip(df.identifier, features_list)}
    else:
        raise ValueError("Invalid feature choice")

    save_features(df, features_dict, args.data_set, args.feature_type)
