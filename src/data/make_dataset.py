# -*- coding: utf-8 -*-
import logging
import sys
import os
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import yaml

project_dir = Path(__file__).resolve().parents[2]


def to_np_array(ak_array, maxN=100, pad=0, dtype=float):
    """convert awkward array to regular numpy array"""
    return ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad).to_numpy().astype(dtype)


@click.command()
@click.argument("definitions", type=click.Path(exists=True), default=f"{project_dir}/src/data/definitions.yml")
@click.option("--train", is_flag=True, show_default=True, default=False)
@click.option("--test", is_flag=True, show_default=True, default=False)
def main(definitions, train, test):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    with open(definitions) as yaml_file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

    spectators = defn["spectators"]
    labels = defn["labels"]
    n_feature_sets = defn["n_feature_sets"]
    batch_size = defn["batch_size"]
    if train:
        dataset = "train"
    elif test:
        dataset = "test"
    else:
        logger.info("You need to specify if they are training/testing dataset by setting --train or --test")
    files = defn[f"{dataset}_files"]

    counter = -1
    for input_file in files:
        # The line below should be deleted after testing
        if counter >= 0:
            sys.exit()
        in_file = uproot.open(input_file)
        tree = in_file[defn["tree_name"]]
        nentries = tree.num_entries
        logger.info(f"opening {input_file} with {nentries} events")
        for k in range(0, nentries, batch_size):
            counter += 1
            if os.path.isfile(f"{project_dir}/data/processed/{dataset}/newdata_{counter}.h5"):
                logger.info(f"{project_dir}/data/processed/{dataset}/newdata_{counter}.h5 exists... skipping")
                continue
            arrays = tree.arrays(spectators, library="np", entry_start=k, entry_stop=k + batch_size)
            spec_array = np.expand_dims(np.stack([arrays[spec] for spec in spectators], axis=1), axis=1)
            real_batch_size = spec_array.shape[0]

            feature_arrays = {}
            for j in range(n_feature_sets):
                feature_arrays[f"features_{j}"] = np.zeros(
                    (real_batch_size, defn[f"nobj_{j}"], len(defn[f"features_{j}"])),
                    dtype=float,
                )
                arrays = tree.arrays(
                    defn[f"features_{j}"],
                    entry_start=k,
                    entry_stop=k + batch_size,
                    library="ak",
                )
                for i, feature in enumerate(defn[f"features_{j}"]):
                    feat = to_np_array(arrays[feature], maxN=defn[f"nobj_{j}"])
                    feature_arrays[f"features_{j}"][:, :, i] = feat
                # For PyTorch channels-first style networks
                feature_arrays[f"features_{j}"] = np.ascontiguousarray(np.swapaxes(feature_arrays[f"features_{j}"], 1, 2))

            arrays = tree.arrays(labels, library="np", entry_start=k, entry_stop=k + batch_size)
            target_array = np.zeros((real_batch_size, 2), dtype=float)
            target_array[:, 0] = arrays["sample_isQCD"] * arrays["fj_isQCD"]
            target_array[:, 1] = arrays["fj_isH"]

            os.makedirs(f"{project_dir}/data/processed/{dataset}", exist_ok=True)
            with h5py.File(f"{project_dir}/data/processed/{dataset}/newdata_{counter}.h5", "w") as h5:
                logger.info(f"creating {h5.filename} h5 file with {real_batch_size} events")
                feature_data = h5.create_group(f"{dataset}ing_subgroup")
                target_data = h5.create_group("target_subgroup")
                # weight_data = h5.create_group("weight_subgroup")
                spec_data = h5.create_group("spectator_subgroup")
                for j in range(n_feature_sets):
                    feature_data.create_dataset(
                        f"{dataset}ing_{j}",
                        data=feature_arrays[f"features_{j}"],
                    )
                target_data.create_dataset("target", data=target_array)
                # weight_data.create_dataset("weights", data=weight_array)
                spec_data.create_dataset("spectators", data=spec_array)
                h5.close()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
