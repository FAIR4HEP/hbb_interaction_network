# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import yaml

project_dir = Path(__file__).resolve().parents[2]
np.random.seed(42)


def to_np_array(ak_array, maxN=100, pad=0, dtype=float):
    """convert awkward array to regular numpy array"""
    return ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad).to_numpy().astype(dtype)


@click.command()
@click.argument(
    "definitions",
    type=click.Path(exists=True),
    default=f"{project_dir}/src/data/definitions.yml",
)
@click.option("--train", is_flag=True, show_default=True, default=False)
@click.option("--test", is_flag=True, show_default=True, default=False)
@click.option("--outdir", show_default=True, default=f"{project_dir}/data/processed/")
@click.option("--max-entries", show_default=True, default=None, type=int)
@click.option("--min-npv", show_default=True, default=-1, type=int)
@click.option("--max-npv", show_default=True, default=9999, type=int)
@click.option("--keep-frac", show_default=True, default=1, type=float)
@click.option("--batch-size", show_default=True, default=None, type=int)
def main(definitions, train, test, outdir, max_entries, min_npv, max_npv, keep_frac, batch_size):  # noqa: C901
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
    if not batch_size:
        batch_size = defn["batch_size"]
    if train:
        dataset = "train"
    elif test:
        dataset = "test"
    else:
        logger.info("You need to specify if they are training/testing dataset by setting --train or --test")
    files = defn[f"{dataset}_files"]

    counter = -1
    total_entries = 0
    done = False
    for input_file in files:
        in_file = uproot.open(input_file)
        tree = in_file[defn["tree_name"]]
        nentries = tree.num_entries
        logger.info(f"opening {input_file} with {nentries} events")
        for k in range(0, nentries, batch_size):
            counter += 1
            if os.path.isfile(f"{outdir}/{dataset}/newdata_{counter}.h5"):
                logger.info(f"{outdir}/{dataset}/newdata_{counter}.h5 exists... skipping")
                continue
            arrays = tree.arrays(spectators, library="np", entry_start=k, entry_stop=k + batch_size)
            mask = (
                (arrays["npv"] >= min_npv) & (arrays["npv"] < max_npv) & (np.random.rand(*arrays["npv"].shape) < keep_frac)
            )
            spec_array = np.expand_dims(np.stack([arrays[spec][mask] for spec in spectators], axis=1), axis=1)
            real_batch_size = spec_array.shape[0]
            total_entries += real_batch_size

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
                    feat = to_np_array(arrays[feature][mask], maxN=defn[f"nobj_{j}"])
                    feature_arrays[f"features_{j}"][:, :, i] = feat
                # For PyTorch channels-first style networks
                feature_arrays[f"features_{j}"] = np.ascontiguousarray(np.swapaxes(feature_arrays[f"features_{j}"], 1, 2))

            arrays = tree.arrays(labels, library="np", entry_start=k, entry_stop=k + batch_size)
            target_array = np.zeros((real_batch_size, 2), dtype=float)
            target_array[:, 0] = arrays["sample_isQCD"][mask] * arrays["fj_isQCD"][mask]
            target_array[:, 1] = arrays["fj_isH"][mask]

            os.makedirs(f"{outdir}/{dataset}", exist_ok=True)
            with h5py.File(f"{outdir}/{dataset}/newdata_{counter}.h5", "w") as h5:
                logger.info(f"creating {h5.filename} h5 file with {real_batch_size} events")
                feature_data = h5.create_group(f"{dataset}ing_subgroup")
                target_data = h5.create_group("target_subgroup")
                spec_data = h5.create_group("spectator_subgroup")
                for j in range(n_feature_sets):
                    feature_data.create_dataset(
                        f"{dataset}ing_{j}",
                        data=feature_arrays[f"features_{j}"].astype("float32"),
                    )
                    np.save(
                        f"{outdir}/{dataset}/{dataset}_{counter}_features_{j}.npy",
                        feature_arrays[f"features_{j}"].astype("float32"),
                    )
                target_data.create_dataset("target", data=target_array.astype("float32"))
                np.save(
                    f"{outdir}/{dataset}/{dataset}_{counter}_truth.npy",
                    target_array.astype("float32"),
                )
                spec_data.create_dataset("spectators", data=spec_array.astype("float32"))
                np.save(
                    f"{outdir}/{dataset}/{dataset}_{counter}_spectators.npy",
                    spec_array.astype("float32"),
                )
                h5.close()
            if max_entries and total_entries >= max_entries:
                done = True
                break
        if done:
            break


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
