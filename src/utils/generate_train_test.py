import argparse
import glob
from pathlib import Path
import os
import sys

import setGPU
import numpy as np
import sklearn.model_selection
import torch
import tqdm
import yaml

sys.path.append("..")
from data.h5data import H5Data  # noqa: E402

print(torch.__version__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

project_dir = Path(__file__).resolve().parents[2]
train_path = f"{project_dir}/data/processed/train/"
definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

N = defn["nobj_2"]  # number of charged particles
N_sv = defn["nobj_3"]  # number of SVs
n_targets = len(defn["reduced_labels"])  # number of classes
device = "cpu"
params = defn["features_2"]
params_sv = defn["features_3"]



"""
#Deep double-b features
params_2 = params_2[22:]
params_3 = params_2[11:13]
"""


def main(args):
    """Main entry point of the app"""
    # Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    
    outdir = args.outdir
    files = glob.glob(train_path + "/newdata_*.h5")
    files_train = files  # take rest for training
    # label = "new"

    data_train = H5Data(
        batch_size=batch_size,
        cache=None,
        preloading=0,
        features_name="training_subgroup",
        labels_name="target_subgroup",
        spectators_name="spectator_subgroup",
    )
    data_train.set_file_names(files_train)

    n_train = data_train.count_data()  # number of all data samples
    batch_size = n_train

    print("train data:", n_train)
    import time

    t_X1 = []
    t_X2 = []
    t_X3 = []
    t_X4 = []
    t_Y = []
    t_Z = []

    # import time
    start_time = time.time()
    for sub_X, sub_Y, sub_Z in tqdm.tqdm(data_train.generate_data(), total=n_train / batch_size):
        t_X1 = sub_X[0]
        t_X2 = sub_X[1]
        t_X3 = sub_X[2]
        t_X4 = sub_X[3]
        t_Y = sub_Y[0]
        t_Z = sub_Z[0]

    end_time = time.time()
    print("time for load data:", end_time - start_time)

    # split using rand
    print(len(t_Z))

    print("splitting test and train!")
    index_list = list(range(len(t_Z)))
    (
        t_X1_tr,
        t_X1_te,
        t_X2_tr,
        t_X2_te,
        t_X3_tr,
        t_X3_te,
        t_X4_tr,
        t_X4_te,
        t_Y_tr,
        t_Y_te,
        t_Z_tr,
        t_Z_te,
    ) = sklearn.model_selection.train_test_split(t_X1, t_X2, t_X3, t_X4, t_Y, t_Z, test_size=0.09, train_size=0.91)
    ind_tr, ind_val = sklearn.model_selection.train_test_split(index_list, test_size=0.09, train_size=0.91)

    print("X1 start")
    t_X1_tr = t_X1[ind_tr]
    t_X1_te = t_X1[ind_val]
    np.save("{}/data_X1_tr.npy".format(outdir), t_X1_tr)
    np.save("{}/data_X1_te.npy".format(outdir), t_X1_te)
    del t_X1
    print("X2")
    t_X2_tr = t_X2[ind_tr]
    t_X2_te = t_X2[ind_val]
    np.save("{}/data_X2_tr.npy".format(outdir), t_X2_tr)
    np.save("{}/data_X2_te.npy".format(outdir), t_X2_te)
    del t_X2
    print("X3")
    t_X3_tr = t_X3[ind_tr]
    t_X3_te = t_X3[ind_val]
    np.save("{}/data_X3_tr.npy".format(outdir), t_X3_tr)
    np.save("{}/data_X3_te.npy".format(outdir), t_X3_te)
    del t_X3
    print("X4")
    t_X4_tr = t_X4[ind_tr]
    t_X4_te = t_X4[ind_val]
    np.save("{}/data_X4_tr.npy".format(outdir), t_X4_tr)
    np.save("{}/data_X4_te.npy".format(outdir), t_X4_te)
    del t_X4
    print("Y")
    t_Y_tr = t_Y[ind_tr]
    t_Y_te = t_Y[ind_val]
    np.save("{}/data_Y_tr.npy".format(outdir), t_Y_tr)
    np.save("{}/data_Y_te.npy".format(outdir), t_Y_te)
    del t_Y
    print("Z")
    t_Z_tr = t_Z[ind_tr]
    t_Z_te = t_Z[ind_val]
    np.save("{}/data_Z_tr.npy".format(outdir), t_Z_tr)
    np.save("{}/data_Z_te.npy".format(outdir), t_Z_te)
    del t_Z
    print("splitting done")

    t_X_tr = [t_X1_tr, t_X2_tr, t_X3_tr, t_X4_tr]
    t_Y_tr = [t_Y_tr]
    t_Z_tr = [t_Z_tr]
    t_X_te = [t_X1_te, t_X2_te, t_X3_te, t_X4_te]
    t_Y_te = [t_Y_te]
    t_Z_te = [t_Z_te]
    del t_X1_tr, t_X2_tr, t_X3_tr, t_X4_tr
    del t_X1_te, t_X2_te, t_X3_te, t_X4_te

    print("byte size for t_X1_tr", np.shape(t_X_tr[0]))  # t_X_tr[0].itemsize*
    print("byte size for t_X2_tr", np.shape(t_X_tr[1]))  # t_X_tr[1].itemsize*
    print("byte size for t_X3_tr", np.shape(t_X_tr[2]))  # t_X_tr[2].itemsize*
    print("byte size for t_X4_tr", np.shape(t_X_tr[3]))  # t_X_tr[3].itemsize*
    print("byte size for t_Y_tr", np.shape(t_Y_tr[0]))  # t_Y_tr[0].itemsize*
    print("byte size for t_Z_tr", np.shape(t_Z_tr[0]))  # t_Z_tr[0].itemsize*
    print("byte size for t_X1_te", np.shape(t_X_te[0]))  # t_X_tr[0].itemsize*
    print("byte size for t_X2_te", np.shape(t_X_te[1]))  # t_X_tr[1].itemsize*
    print("byte size for t_X3_te", np.shape(t_X_te[2]))  # t_X_tr[2].itemsize*
    print("byte size for t_X4_te", np.shape(t_X_te[3]))  # t_X_tr[3].itemsize*
    print("byte size for t_Y_te", np.shape(t_Y_te[0]))  # t_Y_tr[0].itemsize*
    print("byte size for t_Z_te", np.shape(t_Z_te[0]))  # t_Z_tr[0].itemsize*
    print(
        "all done",
        len(t_X_te),
        np.shape(t_X_te[0]),
        np.shape(t_X_te[1]),
        np.shape(t_X_te[2]),
        np.shape(t_X_te[3]),
        len(t_Y_te),
        len(t_Z_te),
    )


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        action="store",
        dest="outdir",
        default="./npy_data2",
        help="Output directory",
    )
    
    args = parser.parse_args()
    main(args)
