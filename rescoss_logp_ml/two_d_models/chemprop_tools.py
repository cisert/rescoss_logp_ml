"""
Modified from chemprop.train.run_training and chemprop.data.scaffold (https://github.com/chemprop/chemprop/, accessed 25.04.22)

The license from the original repo is reproduced here as requested: 

MIT License

Copyright (c) 2020 Wengong Jin, Kyle Swanson, Kevin Yang, Regina Barzilay, Tommi Jaakkola

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
from logging import Logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # needed for torch (in chemprop)
torch.set_num_threads(int(os.environ["OPENBLAS_NUM_THREADS"]))
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

from chemprop.train.evaluate import evaluate, evaluate_predictions
from chemprop.train.predict import predict
from chemprop.train.train import train
from chemprop.train.loss_functions import get_loss_func
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import (
    build_optimizer,
    build_lr_scheduler,
    load_checkpoint,
    makedirs,
    save_checkpoint,
    save_smiles_splits,
    load_frzn_model,
)
from rescoss_logp_ml.utils import SEED
from random import Random
from typing import Dict, List, Set, Tuple, Union
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


def run_training(
    args: TrainArgs, data: MoleculeDataset, logger: Logger = None, val_at_epoch=None
) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    from tensorboardX import SummaryWriter

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    debug(f"Splitting data with seed {args.seed}")
    if args.separate_test_path:
        test_data = get_data(
            path=args.separate_test_path,
            args=args,
            features_path=args.separate_test_features_path,
            atom_descriptors_path=args.separate_test_atom_descriptors_path,
            bond_features_path=args.separate_test_bond_features_path,
            phase_features_path=args.separate_test_phase_features_path,
            smiles_columns=args.smiles_columns,
            loss_function=args.loss_function,
            logger=logger,
        )
    if args.separate_val_path:
        val_data = get_data(
            path=args.separate_val_path,
            args=args,
            features_path=args.separate_val_features_path,
            atom_descriptors_path=args.separate_val_atom_descriptors_path,
            bond_features_path=args.separate_val_bond_features_path,
            phase_features_path=args.separate_val_phase_features_path,
            smiles_columns=args.smiles_columns,
            loss_function=args.loss_function,
            logger=logger,
        )

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
    else:
        train_data, val_data, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )

    if args.dataset_type == "classification":
        class_sizes = get_class_sizes(data)
        debug("Class sizes")
        for i, task_class_sizes in enumerate(class_sizes):
            debug(
                f"{args.task_names[i]} "
                f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}'
            )

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_features=True)
        val_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
        test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None

    args.train_data_size = len(train_data)

    debug(
        f"Total size = {len(data):,} | "
        f"train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}"
    )

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == "regression":
        debug("Fitting scaler")
        scaler = train_data.normalize_targets()
    elif args.dataset_type == "spectra":
        debug("Normalizing spectra and excluding spectra regions based on phase")
        args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
        for dataset in [train_data, test_data, val_data]:
            data_targets = normalize_spectra(
                spectra=dataset.targets(),
                phase_features=dataset.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=None,
                threshold=args.spectra_target_floor,
            )
            dataset.set_targets(data_targets)
        scaler = None
    else:
        scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == "multiclass":
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed,
    )
    val_data_loader = MoleculeDataLoader(dataset=val_data, batch_size=args.batch_size, num_workers=num_workers)
    test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=num_workers)

    if args.class_balance:
        debug(f"With class_balance, effective train size = {train_data_loader.iter_size:,}")

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f"Loading model {model_idx} from {args.checkpoint_paths[model_idx]}")
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f"Building model {model_idx}")
            model = MoleculeModel(args)

        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            debug(f"Loading and freezing parameters from {args.checkpoint_frzn}.")
            model = load_frzn_model(model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

        debug(model)

        if args.checkpoint_frzn is not None:
            debug(f"Number of unfrozen parameters = {param_count(model):,}")
            debug(f"Total number of parameters = {param_count_all(model):,}")
        else:
            debug(f"Number of parameters = {param_count_all(model):,}")

        if args.cuda:
            debug("Moving model to cuda")
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(
            os.path.join(save_dir, MODEL_FILE_NAME),
            model,
            scaler,
            features_scaler,
            atom_descriptor_scaler,
            bond_feature_scaler,
            args,
        )

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float("inf") if args.minimize_score else -float("inf")
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f"Epoch {epoch}")
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            if val_at_epoch is None:
                validate_now = True  # validate at each epoch
            else:
                if epoch == val_at_epoch:
                    validate_now = True  # we want to validate at a specific epoch, and that epoch is now
                else:
                    validate_now = False  # we want to validate at a specific epoch, but that epoch is not now
            if validate_now:
                val_scores = evaluate(
                    model=model,
                    data_loader=val_data_loader,
                    num_tasks=args.num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    logger=logger,
                )

                for metric, scores in val_scores.items():
                    # Average validation score
                    avg_val_score = np.nanmean(scores)
                    debug(f"Validation {metric} = {avg_val_score:.6f}")
                    writer.add_scalar(f"validation_{metric}", avg_val_score, n_iter)

                    if args.show_individual_scores:
                        # Individual validation scores
                        for task_name, val_score in zip(args.task_names, scores):
                            debug(f"Validation {task_name} {metric} = {val_score:.6f}")
                            writer.add_scalar(f"validation_{task_name}_{metric}", val_score, n_iter)

                # Save model checkpoint if improved validation score
                avg_val_score = np.nanmean(val_scores[args.metric])

                if (
                    args.minimize_score
                    and avg_val_score < best_score
                    or not args.minimize_score
                    and avg_val_score > best_score
                ):
                    best_score, best_epoch = avg_val_score, epoch
                    best_scores = {}
                    for metric in args.metrics:
                        avr_metric = np.nanmean(val_scores[metric])
                        best_scores[metric] = avr_metric
                    save_checkpoint(
                        os.path.join(save_dir, MODEL_FILE_NAME),
                        model,
                        scaler,
                        features_scaler,
                        atom_descriptor_scaler,
                        bond_feature_scaler,
                        args,
                    )

        # Evaluate on test/validation set
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        test_preds = predict(model=model, data_loader=test_data_loader, scaler=scaler)

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            gt_targets=test_data.gt_targets(),
            lt_targets=test_data.lt_targets(),
            logger=logger,
        )
        test_preds = np.asarray(test_preds).flatten()

        writer.close()
    model_savepath = os.path.join(save_dir, MODEL_FILE_NAME)
    return best_scores, best_epoch, test_scores, test_preds, model_savepath


def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, tuple):
        mol = mol[0]
    elif isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        raise ValueError
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(
    mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]], use_indices: bool = False
) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(
    data: MoleculeDataset, sizes: Tuple[float], key_molecule_index: int = 0, seed: int = SEED,
) -> Tuple[MoleculeDataset]:
    r"""
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.
    :param data: A :class:`MoleculeDataset`.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not np.isclose(sum(sizes), 1):
        raise ValueError(f"Invalid splits! got: {sizes}")

    # Split
    set_sizes = [int(s * len(data)) for s in sizes[:-1]]
    set_sizes.append(len(data) - sum(set_sizes))  # to make sure we're not losing anything due to rounding
    idxs = [[] for _ in sizes]
    counts = [0 for _ in sizes]  # number of different scaffolds in individual splits
    mols_per_scaffold = [[] for _ in sizes]

    # Map from scaffold to index in the data
    key_mols = [m[key_molecule_index] for m in data.mols(flatten=False)]
    scaffold_to_indices = scaffold_to_smiles(key_mols, use_indices=True)
    scaffold_to_mols = scaffold_to_smiles(key_mols, use_indices=False)

    # Seed randomness
    random = Random(seed)

    index_sets = list(scaffold_to_indices.values())
    random.seed(seed)
    random.shuffle(index_sets)

    i = 0  # counter for index_sets
    j = 0  # counter for splits
    while j <= len(sizes) - 1:
        while i <= len(index_sets) - 1:  # continue until we have used all index_sets
            if len(idxs[j]) + len(index_sets[i]) <= set_sizes[j]:
                idxs[j].extend(list(index_sets[i]))
                counts[j] += 1
                mols_per_scaffold[j].append(len(index_sets[i]))
                i += 1
            else:
                if j == len(sizes) - 1:  # last fold --> add anyways
                    while i <= len(index_sets) - 1:  # add remaining index_sets
                        idxs[j] += index_sets[i]
                        counts[j] += 1
                        mols_per_scaffold[j].append(len(index_sets[i]))
                        i += 1
                    j += 1  # to break the loop
                    break
                else:  # not last fold: move onto next one
                    j += 1
        j += 1

    # Map from indices to data
    data_sets = []
    for idx in idxs:
        data_set = MoleculeDataset([data[i] for i in idx])
        data_sets.append(data_set)

    return data_sets, idxs, counts, mols_per_scaffold
