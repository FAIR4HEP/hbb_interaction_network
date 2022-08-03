from __future__ import print_function

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
train_path = "../../data/processed/train"

N = 60  # number of charged particles
N_sv = 5  # number of SVs
n_targets = 2  # number of classes
device = "cpu"

params = [
    "track_ptrel",
    "track_erel",
    "track_phirel",
    "track_etarel",
    "track_deltaR",
    "track_drminsv",
    "track_drsubjet1",
    "track_drsubjet2",
    "track_dz",
    "track_dzsig",
    "track_dxy",
    "track_dxysig",
    "track_normchi2",
    "track_quality",
    "track_dptdpt",
    "track_detadeta",
    "track_dphidphi",
    "track_dxydxy",
    "track_dzdz",
    "track_dxydz",
    "track_dphidxy",
    "track_dlambdadz",
    "trackBTag_EtaRel",
    "trackBTag_PtRatio",
    "trackBTag_PParRatio",
    "trackBTag_Sip2dVal",
    "trackBTag_Sip2dSig",
    "trackBTag_Sip3dVal",
    "trackBTag_Sip3dSig",
    "trackBTag_JetDistVal",
]

params_sv = [
    "sv_ptrel",
    "sv_erel",
    "sv_phirel",
    "sv_etarel",
    "sv_deltaR",
    "sv_pt",
    "sv_mass",
    "sv_ntracks",
    "sv_normchi2",
    "sv_dxy",
    "sv_dxysig",
    "sv_d3d",
    "sv_d3dsig",
    "sv_costhetasvpv",
]


def main(args):
    """Main entry point of the app"""

    from src.data.h5data import H5Data

    files = glob.glob(train_path + "/newdata_*.h5")
    files_val = files[:5]  # take first 5 for validation
    files_train = files[5:]  # take rest for training

    label = "new"
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    batch_size = 128
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

    from models import INTagger

    gnn = INTagger(
        N,
        N_sv,
        n_targets,
        len(params),
        len(params_sv),
        args.hidden,
        De=args.De,
        Do=args.Do,
    )

    n_epochs = 200
    patience = 8

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

    for m in range(n_epochs):
        final_epoch = m
        lst = []
        loss_val = []
        loss_training = []
        correct = []
        for sub_X, sub_Y, _ in tqdm.tqdm(data_train.generate_data(), total=n_train / batch_size):
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]
            trainingv = (torch.FloatTensor(training)).to(device)
            trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
            targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)
            optimizer.zero_grad()

            # Input training dataset
            out = gnn(trainingv.to(device), trainingv_sv.to(device))

            l_train = loss(out, targetv.to(device))
            loss_training.append(l_train.item())
            l_train.backward()
            optimizer.step()
            del trainingv, trainingv_sv, targetv
        for sub_X, sub_Y, _ in tqdm.tqdm(data_val.generate_data(), total=n_val / batch_size):
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]
            trainingv = (torch.FloatTensor(training)).to(device)
            trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
            targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)

            # Input validation dataset
            out = gnn(trainingv.to(device), trainingv_sv.to(device))

            lst.append(softmax(out).cpu().data.numpy())
            l_val = loss(out, targetv.to(device))
            loss_val.append(l_val.item())

            correct.append(target)
            del trainingv, trainingv_sv, targetv
        l_val = np.mean(np.array(loss_val))

        predicted = np.concatenate(lst)

        l_training = np.mean(np.array(loss_training))
        val_targetv = np.concatenate(correct)

        torch.save(gnn.state_dict(), "%s/gnn_%s_last.pth" % (outdir, label))
        if l_val < l_val_best:
            l_val_best = l_val
            torch.save(gnn.state_dict(), "%s/gnn_%s_best.pth" % (outdir, label))

        acc_vals_validation[m] = accuracy_score(val_targetv[:, 0], predicted[:, 0] > 0.5)
        loss_vals_training[m] = l_training
        loss_vals_validation[m] = l_val
        loss_std_validation[m] = np.std(np.array(loss_val))
        loss_std_training[m] = np.std(np.array(loss_training))
        if m > patience and all(
            loss_vals_validation[max(0, m - patience) : m]
            > min(np.append(loss_vals_validation[0 : max(0, m - patience)], n_epochs))
        ):
            break

    acc_vals_validation = acc_vals_validation[: (final_epoch + 1)]
    loss_vals_training = loss_vals_training[: (final_epoch + 1)]
    loss_vals_validation = loss_vals_validation[: (final_epoch + 1)]
    loss_std_validation = loss_std_validation[: (final_epoch + 1)]
    loss_std_training = loss_std_training[:(final_epoch)]
    np.save("%s/acc_vals_validation_%s.npy" % (outdir, label), acc_vals_validation)
    np.save("%s/loss_vals_training_%s.npy" % (outdir, label), loss_vals_training)
    np.save("%s/loss_vals_validation_%s.npy" % (outdir, label), loss_vals_validation)
    np.save("%s/loss_std_validation_%s.npy" % (outdir, label), loss_std_validation)
    np.save("%s/loss_std_training_%s.npy" % (outdir, label), loss_std_training)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")

    # Optional arguments
    parser.add_argument("--De", type=int, action="store", dest="De", default=20, help="De")
    parser.add_argument("--Do", type=int, action="store", dest="Do", default=24, help="Do")
    parser.add_argument("--hidden", type=int, action="store", dest="hidden", default=60, help="hidden")

    args = parser.parse_args()
    main(args)
