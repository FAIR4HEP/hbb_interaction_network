from __future__ import print_function

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
if torch.cuda.is_available():
    import setGPU
import tqdm
import yaml

from src.data.h5data import H5Data
from src.models.models import GraphNet

# import sys
# sys.path.append("..")
# from data.h5data import H5Data     # noqa: E402
# from models import GraphNet  # noqa: E402

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

project_dir = Path(__file__).resolve().parents[2]

train_path = f"{project_dir}/data/processed/train/"
definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

N = defn["nobj_2"]  # number of charged particles
N_sv = defn["nobj_3"]  # number of SVs
n_targets = len(defn["reduced_labels"])  # number of classes
params = defn["features_2"]
params_sv = defn["features_3"]


def main(args):  # noqa: C901
    """Main entry point of the app"""
    model_dict = {}

    device = args.device

    files = glob.glob(os.path.join(train_path, "newdata_*.h5"))
    files_val = files[:5]  # take first 5 for validation
    files_train = files[5:]  # take rest for training

    outdir = args.outdir
    vv_branch = args.vv_branch
    drop_rate = args.drop_rate
    load_def = args.load_def
    random_split = args.random_split
    indir = args.indir

    if args.drop_pfeatures != "":
        drop_pfeatures = list(map(int, str(args.drop_pfeatures).split(",")))
    else:
        drop_pfeatures = []

    if args.drop_svfeatures != "":
        drop_svfeatures = list(map(int, str(args.drop_svfeatures).split(",")))
    else:
        drop_svfeatures = []

    label = args.label
    if label == "" and drop_rate != 0:
        label = "new_DR" + str(int(drop_rate * 100.0))
    elif label == "" and drop_rate == 0:
        label = "new"
    if len(drop_pfeatures) > 0:
        print("The following particle candidate features to be dropped: ", drop_pfeatures)
    if len(drop_svfeatures) > 0:
        print("The following secondary vertex features to be dropped: ", drop_svfeatures)
    n_epochs = args.epoch
    batch_size = args.batch_size
    model_loc = "{}/trained_models/".format(outdir)
    model_perf_loc = "{}/model_performances".format(outdir)
    model_dict_loc = "{}/model_dicts".format(outdir)
    os.system("mkdir -p {} {} {}".format(model_loc, model_perf_loc, model_dict_loc))

    # Saving the model's metadata as a json dict
    for arg in vars(args):
        model_dict[arg] = getattr(args, arg)
    f_model = open("{}/gnn_{}_model_metadata.json".format(model_dict_loc, label), "w")
    json.dump(model_dict, f_model, indent=3)
    f_model.close()

    # Get the training and validation data
    if random_split is False:
        data_train = H5Data(
            batch_size=batch_size,
            cache=None,
            preloading=0,
            features_name="training_subgroup",
            labels_name="target_subgroup",
            spectators_name="spectator_subgroup",
        )
        data_train.set_file_names(files_train)
        data_val = H5Data(
            batch_size=batch_size,
            cache=None,
            preloading=0,
            features_name="training_subgroup",
            labels_name="target_subgroup",
            spectators_name="spectator_subgroup",
        )
        data_val.set_file_names(files_val)

        n_val = data_val.count_data()
        n_train = data_train.count_data()
        print("val data:", n_val)
        print("train data:", n_train)
    else:
        t_X1_tr = np.load("{}/data_X1_tr.npy".format(indir))
        t_X2_tr = np.load("{}/data_X2_tr.npy".format(indir))
        t_X3_tr = np.load("{}/data_X3_tr.npy".format(indir))

        t_X4_tr = np.load("{}/data_X4_tr.npy".format(indir))
        t_Y_tr = np.load("{}/data_Y_tr.npy".format(indir))
        t_Z_tr = np.load("{}/data_Z_tr.npy".format(indir))

        t_X1_te = np.load("{}/data_X1_te.npy".format(indir))
        t_X2_te = np.load("{}/data_X2_te.npy".format(indir))
        t_X3_te = np.load("{}/data_X3_te.npy".format(indir))

        t_X4_te = np.load("{}/data_X4_te.npy".format(indir))
        t_Y_te = np.load("{}/data_Y_te.npy".format(indir))
        t_Z_te = np.load("{}/data_Z_te.npy".format(indir))

        # print("seperate converting finished")
        t_X_tr = [t_X1_tr, t_X2_tr, t_X3_tr, t_X4_tr]
        t_Y_tr = [t_Y_tr]
        t_Z_tr = [t_Z_tr]
        print("mid for train finish numpy convert")
        t_X_te = [t_X1_te, t_X2_te, t_X3_te, t_X4_te]
        t_Y_te = [t_Y_te]
        t_Z_te = [t_Z_te]

        del t_X1_tr, t_X2_tr, t_X3_tr, t_X4_tr
        del t_X1_te, t_X2_te, t_X3_te, t_X4_te

        print("!!!", len(t_X_te), len(t_Y_te), len(t_Z_te))

    gnn = GraphNet(
        n_constituents=N,
        n_targets=n_targets,
        params=len(params) - len(drop_pfeatures),
        hidden=args.hidden,
        n_vertices=N_sv,
        params_v=len(params_sv) - len(drop_svfeatures),
        vv_branch=int(vv_branch),
        De=args.De,
        Do=args.Do,
        device=device,
    )

    """
         N = Number of charged particles (60)
         n_targets = 2 (number of target_class)
         hidden = number of nodes in hidden layers
         params = number of features for each charged particle (30)
         n_vertices = number of secondary vertices (5)
         params_v = number of features for secondary vertices (14)
         vv_branch = to allow vv_branch ? (0 or False by default)
         De = Output dimension of particle-particle interaction NN (fR)
         Do = Output dimension of pre-aggregator transformation NN (fO)
         device = device to train gnn; follow pytorch convention
    """

    if load_def:
        if os.path.exists("../../models/trained_models/gnn_baseline_best.pth"):
            defmodel_exists = True
        else:
            defmodel_exists = False
        if not defmodel_exists:
            print("Default model not found, skipping model preloading")
            load_def = False

    if load_def:
        def_state_dict = torch.load("../../models/trained_models/gnn_baseline_best.pth")
        new_state_dict = gnn.state_dict()
        for key in def_state_dict.keys():
            if key not in ["fr1_pv.weight", "fr1.weight", "fo1.weight"]:
                if new_state_dict[key].shape != def_state_dict[key].shape:
                    print(
                        "Tensor shapes don't match for key='{}': old = ({},{}); new = ({},{}): not updating it".format(
                            key,
                            def_state_dict[key].shape[0],
                            def_state_dict[key].shape[1],
                            new_state_dict[key].shape[0],
                            new_state_dict[key].shape[1],
                        )
                    )
                else:
                    new_state_dict[key] = def_state_dict[key].clone()
            else:
                if key == "fr1_pv.weight":
                    indices_to_keep = [i for i in range(len(params)) if i not in drop_pfeatures] + [
                        len(params) + i for i in range(len(params_sv)) if i not in drop_svfeatures
                    ]

                if key == "fr1.weight":
                    indices_to_keep = [i for i in range(len(params)) if i not in drop_pfeatures] + [
                        len(params) + i for i in range(len(params)) if i not in drop_pfeatures
                    ]

                if key == "fo1.weight":
                    indices_to_keep = [i for i in range(len(params)) if i not in drop_pfeatures] + list(
                        range(len(params), len(params) + 2 * args.De)
                    )

                new_tensor = def_state_dict[key][:, indices_to_keep]
                if new_state_dict[key].shape != new_tensor.shape:
                    print(
                        "Tensor shapes don't match for "
                        + "key='{}': modified old = ({},{}); new = ({},{}): not updating it".format(
                            key,
                            new_tensor.shape[0],
                            new_tensor.shape[1],
                            new_state_dict[key].shape[0],
                            new_state_dict[key].shape[1],
                        )
                    )

                else:
                    new_state_dict[key] = new_tensor.clone()
        gnn.load_state_dict(new_state_dict)

    loss = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(gnn.parameters(), lr=0.0001)
    loss_vals_training = np.zeros(n_epochs)
    loss_std_training = np.zeros(n_epochs)
    loss_vals_validation = np.zeros(n_epochs)
    loss_std_validation = np.zeros(n_epochs)

    acc_vals_validation = np.zeros(n_epochs)

    final_epoch = 0
    l_val_best = 99999

    from sklearn.metrics import accuracy_score

    softmax = torch.nn.Softmax(dim=1)
    import time

    for m in range(n_epochs):
        print("Epoch %s\n" % m)
        final_epoch = m
        lst = []
        loss_val = []
        loss_training = []
        correct = []
        tic = time.perf_counter()
        sig_count = 0
        data_dropped = 0

        # train process
        if random_split is False:
            iterator = data_train.generate_data()
            total_ = int(n_train / batch_size)
        else:
            batch_num_tr = int(len(t_X_tr[1]) / batch_size)
            print("batch num, X_tr_1, batch_size: ", batch_num_tr, len(t_X_tr[1]), batch_size)
            iterator = range(batch_num_tr)
            total_ = batch_num_tr

        for element in tqdm.tqdm(iterator, total=total_):
            if random_split is False:
                (sub_X, sub_Y, _) = element
                training = sub_X[2]
                training_sv = sub_X[3]
                target = sub_Y[0]

                trainingv = (torch.FloatTensor(training)).to(device)
                trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
                targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)
                
            else:
                idx_ = element
                if idx_ == batch_num_tr - 1:
                    training = t_X_tr[2][idx_ * batch_size : -1]
                    training_sv = t_X_tr[3][idx_ * batch_size : -1]
                    target = t_Y_tr[0][idx_ * batch_size : -1]

                    trainingv = (torch.FloatTensor(training)).to(device)
                    trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
                    targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)

                else:
                    training = t_X_tr[2][idx_ * batch_size : (idx_ + 1) * batch_size]
                    training_sv = t_X_tr[3][idx_ * batch_size : (idx_ + 1) * batch_size]
                    target = t_Y_tr[0][idx_ * batch_size : (idx_ + 1) * batch_size]

                    trainingv = (torch.FloatTensor(training)).to(device)
                    trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
                    targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)

            if drop_rate > 0:
                keep_indices = targetv == 0
                sig_count += batch_size - torch.sum(keep_indices).item()
                to_turn_off = int((batch_size - torch.sum(keep_indices).item()) * drop_rate)
                psum = torch.cumsum(~keep_indices, 0)
                to_keep_after = torch.sum(psum <= to_turn_off).item()
                keep_indices[to_keep_after:] = torch.Tensor([True] * (batch_size - to_keep_after))
                data_dropped += batch_size - torch.sum(keep_indices).item()
                trainingv = trainingv[keep_indices]
                trainingv_sv = trainingv_sv[keep_indices]
                targetv = targetv[keep_indices]
            if len(drop_pfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params), 1, dtype=int) if i not in drop_pfeatures]
                trainingv = trainingv[:, keep_features, :]
            if len(drop_svfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params_sv), 1, dtype=int) if i not in drop_svfeatures]
                trainingv_sv = trainingv_sv[:, keep_features, :]

            optimizer.zero_grad()
            out = gnn(trainingv.to(device), trainingv_sv.to(device))
            batch_loss = loss(out, targetv.to(device))
            loss_training.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()
            del trainingv, trainingv_sv, targetv

        if drop_rate > 0.0:
            print("Signal Count: {}, Data Dropped: {}".format(sig_count, data_dropped))
        toc = time.perf_counter()
        print(f"Training done in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()

        # validate process
        if random_split is False:
            iterator = data_val.generate_data()
            total_ = int(n_val / batch_size)
        else:
            batch_num_te = int(len(t_X_te[1]) / batch_size)
            print("batch num, X_te_1, batch_size: ", batch_num_te, len(t_X_te[1]), batch_size)
            iterator = range(batch_num_te)
            total_ = batch_num_te

        for element in tqdm.tqdm(iterator, total=total_):
            if random_split is False:
                (sub_X, sub_Y, _) = element
                training = sub_X[2]
                training_sv = sub_X[3]
                target = sub_Y[0]

                trainingv = (torch.FloatTensor(training)).to(device)
                trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
                targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)

            else:
                idx_ = element
                if idx_ == batch_num_tr - 1:
                    training = t_X_tr[2][idx_ * batch_size : -1]
                    training_sv = t_X_tr[3][idx_ * batch_size : -1]
                    target = t_Y_tr[0][idx_ * batch_size : -1]

                    trainingv = (torch.FloatTensor(training)).to(device)
                    trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
                    targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)

                else:
                    training = t_X_tr[2][idx_ * batch_size : (idx_ + 1) * batch_size]
                    training_sv = t_X_tr[3][idx_ * batch_size : (idx_ + 1) * batch_size]
                    target = t_Y_tr[0][idx_ * batch_size : (idx_ + 1) * batch_size]

                    trainingv = (torch.FloatTensor(training)).to(device)
                    trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
                    targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)

            if len(drop_pfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params), 1, dtype=int) if i not in drop_pfeatures]
                trainingv = trainingv[:, keep_features, :]
            if len(drop_svfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params_sv), 1, dtype=int) if i not in drop_svfeatures]
                trainingv_sv = trainingv_sv[:, keep_features, :]

            out = gnn(trainingv.to(device), trainingv_sv.to(device))

            lst.append(softmax(out).cpu().data.numpy())
            l_val = loss(out, targetv.to(device))
            loss_val.append(l_val.item())

            correct.append(target)
            del trainingv, trainingv_sv, targetv
        toc = time.perf_counter()
        print(f"Evaluation done in {toc - tic:0.4f} seconds")
        l_val = np.mean(np.array(loss_val))

        predicted = np.concatenate(lst)
        print("\nValidation Loss: ", l_val)

        l_training = np.mean(np.array(loss_training))
        print("Training Loss: ", l_training)
        val_targetv = np.concatenate(correct)

        torch.save(gnn.state_dict(), "%s/gnn_%s_last.pth" % (model_loc, label))
        if l_val < l_val_best:
            print("new best model")
            l_val_best = l_val
            torch.save(gnn.state_dict(), "%s/gnn_%s_best.pth" % (model_loc, label))
            np.save(
                "%s/validation_target_vals_%s.npy" % (model_perf_loc, label),
                val_targetv,
            )
            np.save(
                "%s/validation_predicted_vals_%s.npy" % (model_perf_loc, label),
                predicted,
            )

        print(val_targetv.shape, predicted.shape)
        print(val_targetv, predicted)
        acc_vals_validation[m] = accuracy_score(val_targetv[:, 0], predicted[:, 0] > 0.5)
        print("Validation Accuracy: ", acc_vals_validation[m])
        loss_vals_training[m] = l_training
        loss_vals_validation[m] = l_val
        loss_std_validation[m] = np.std(np.array(loss_val))
        loss_std_training[m] = np.std(np.array(loss_training))
        if m > 8 and all(
            loss_vals_validation[max(0, m - 8) : m] > min(np.append(loss_vals_validation[0 : max(0, m - 8)], 200))
        ):
            print("Early Stopping...")
            print(loss_vals_training, "\n", np.diff(loss_vals_training))
            break
        print()

    acc_vals_validation = acc_vals_validation[: (final_epoch + 1)]
    loss_vals_training = loss_vals_training[: (final_epoch + 1)]
    loss_vals_validation = loss_vals_validation[: (final_epoch + 1)]
    loss_std_validation = loss_std_validation[: (final_epoch + 1)]
    loss_std_training = loss_std_training[:(final_epoch)]
    np.save("%s/acc_vals_validation_%s.npy" % (model_perf_loc, label), acc_vals_validation)
    np.save("%s/loss_vals_training_%s.npy" % (model_perf_loc, label), loss_vals_training)
    np.save("%s/loss_vals_validation_%s.npy" % (model_perf_loc, label), loss_vals_validation)
    np.save("%s/loss_std_validation_%s.npy" % (model_perf_loc, label), loss_std_validation)
    np.save("%s/loss_std_training_%s.npy" % (model_perf_loc, label), loss_std_training)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        action="store",
        dest="outdir",
        default="../../models/",
        help="Output directory",
    )
    parser.add_argument(
        "--npy_indir",
        type=str,
        action="store",
        dest="indir",
        default="./npy_data2",
        help="Output directory",
    )
    parser.add_argument(
        "--vv_branch",
        action="store_true",
        dest="vv_branch",
        default=False,
        help="Consider vertex-vertex interaction in model",
    )

    parser.add_argument("--De", type=int, action="store", dest="De", default=20, help="De")
    parser.add_argument("--Do", type=int, action="store", dest="Do", default=24, help="Do")
    parser.add_argument("--hidden", type=int, action="store", dest="hidden", default=60, help="hidden")
    parser.add_argument(
        "--drop-rate",
        type=float,
        action="store",
        dest="drop_rate",
        default=0.0,
        help="Signal Drop rate",
    )
    parser.add_argument("--epoch", type=int, action="store", dest="epoch", default=100, help="Epochs")
    parser.add_argument(
        "--drop-pfeatures",
        type=str,
        action="store",
        dest="drop_pfeatures",
        default="",
        help="comma separated indices of the particle candidate features to be dropped",
    )
    parser.add_argument(
        "--drop-svfeatures",
        type=str,
        action="store",
        dest="drop_svfeatures",
        default="",
        help="comma separated indices of the secondary vertex features to be dropped",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="",
        help="a label for the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        dest="batch_size",
        default=128,
        help="batch_size",
    )
    parser.add_argument(
        "--load-def",
        action="store_true",
        dest="load_def",
        default=False,
        help="Load weights from default model if enabled",
    )
    parser.add_argument(
        "--random_split",
        action="store_true",
        dest="random_split",
        default=False,
        help="randomly split train test data if enabled",
    )
    parser.add_argument(
        "--device", action="store", dest="device", default="cpu", help="device to train gnn; follow pytorch convention"
    )


    args = parser.parse_args()
    main(args)
