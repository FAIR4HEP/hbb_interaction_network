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
    import setGPU  # noqa: F401

import tqdm
import yaml

from src.data.h5data import H5Data
from src.models.InteractionNet import InteractionNetSingleTagger, InteractionNetTagger
from src.models.pretrain_vicreg import Projector, VICReg, get_backbones

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

project_dir = Path(__file__).resolve().parents[2]

definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

N = defn["nobj_2"]  # number of charged particles
N_sv = defn["nobj_3"]  # number of SVs
n_targets = len(defn["reduced_labels"])  # number of classes
params = defn["features_2"]
params_sv = defn["features_3"]


def main(args):  # noqa: C901

    device = args.device

    files = glob.glob(os.path.join(args.train_path, "newdata_*.h5"))
    # take first 10% of files for validation
    # n_val should be 5 for full dataset
    n_val = max(1, int(0.1 * len(files)))
    files_val = files[:n_val]
    files_train = files[n_val:]

    outdir = args.outdir
    just_svs = args.just_svs
    just_tracks = args.just_tracks

    label = args.label
    if just_svs:
        label += "_just_svs"
    if just_tracks:
        label += "_just_tracks"
    n_epochs = args.epoch
    batch_size = args.batch_size
    model_loc = f"{outdir}/trained_models/"
    model_perf_loc = f"{outdir}/model_performances"
    model_dict_loc = f"{outdir}/model_dicts"
    os.system(f"mkdir -p {model_loc} {model_perf_loc} {model_dict_loc}")

    # Saving the model's metadata as a json dict
    model_dict = {}
    for arg in vars(args):
        model_dict[arg] = getattr(args, arg)
    f_model = open(f"{model_dict_loc}/gnn_{label}_model_metadata.json", "w")
    json.dump(model_dict, f_model, indent=3)
    f_model.close()

    # Get the training and validation data
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
    print(f"val data: {n_val}")
    print(f"train data: {n_train}")

    if args.load_vicreg_path:
        args.x_inputs = len(params)
        args.y_inputs = len(params_sv)
        args.x_backbone, args.y_backbone = get_backbones(args)
        args.return_embedding = False
        args.return_representations = True
        model = VICReg(args).to(args.device)
        model.load_state_dict(torch.load(args.load_vicreg_path))
        model.eval()
        projector = Projector(args.finetune_mlp, 2 * model.Do).to(args.device)
        optimizer = optim.Adam(projector.parameters(), lr=0.0001)
    else:
        if just_svs:
            gnn = InteractionNetSingleTagger(
                dims=N_sv,
                num_classes=n_targets,
                features_dims=len(params_sv),
                hidden=args.hidden,
                De=args.De,
                Do=args.Do,
            ).to(device)
        elif just_tracks:
            gnn = InteractionNetSingleTagger(
                dims=N,
                num_classes=n_targets,
                features_dims=len(params),
                hidden=args.hidden,
                De=args.De,
                Do=args.Do,
            ).to(device)
        else:
            gnn = InteractionNetTagger(
                pf_dims=N,
                sv_dims=N_sv,
                num_classes=n_targets,
                pf_features_dims=len(params),
                sv_features_dims=len(params_sv),
                hidden=args.hidden,
                De=args.De,
                Do=args.Do,
            ).to(device)
        optimizer = optim.Adam(gnn.parameters(), lr=0.0001)

    loss = nn.CrossEntropyLoss(reduction="mean")

    l_val_best = 99999

    from sklearn.metrics import accuracy_score

    softmax = torch.nn.Softmax(dim=1)
    import time

    for m in range(n_epochs):
        print(f"Epoch {m}\n")
        lst = []
        loss_val = []
        loss_train = []
        correct = []
        tic = time.perf_counter()

        # train process
        iterator = data_train.generate_data()
        total_ = int(n_train / batch_size)

        pbar = tqdm.tqdm(iterator, total=total_)
        for element in pbar:
            (sub_X, sub_Y, _) = element
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]

            trainingv = (torch.FloatTensor(training)).to(device)
            trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
            targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)
            optimizer.zero_grad()
            if args.load_vicreg_path:
                projector.train()
                representation, representation_sv = model(trainingv, trainingv_sv)
                out = projector(torch.cat((representation, representation_sv), dim=-1))
            else:
                gnn.train()
                if just_svs:
                    out = gnn(trainingv_sv)
                elif just_tracks:
                    out = gnn(trainingv)
                else:
                    out = gnn(trainingv, trainingv_sv)
            batch_loss = loss(out, targetv)
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.detach().cpu().item()
            loss_train.append(batch_loss)
            pbar.set_description(f"Training loss: {batch_loss:.4f}")

        toc = time.perf_counter()
        print(f"Training done in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()

        # validate process
        iterator = data_val.generate_data()
        total_ = int(n_val / batch_size)
        pbar = tqdm.tqdm(iterator, total=total_)
        for element in pbar:
            (sub_X, sub_Y, _) = element
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]

            trainingv = (torch.FloatTensor(training)).to(device)
            trainingv_sv = (torch.FloatTensor(training_sv)).to(device)
            targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)
            if args.load_vicreg_path:
                projector.eval()
                embedding, embedding_sv = model(trainingv, trainingv_sv)
                out = projector(torch.cat((embedding, embedding_sv), dim=-1))
            else:
                gnn.eval()
                if just_svs:
                    out = gnn(trainingv_sv)
                elif just_tracks:
                    out = gnn(trainingv)
                else:
                    out = gnn(trainingv, trainingv_sv)
            lst.append(softmax(out).cpu().data.numpy())
            l_val = loss(out, targetv).cpu().item()
            loss_val.append(l_val)
            correct.append(target)
            pbar.set_description(f"Validation loss: {l_val:.4f}")
        toc = time.perf_counter()
        print(f"Evaluation done in {toc - tic:0.4f} seconds")
        l_val = np.mean(np.array(loss_val))

        predicted = np.concatenate(lst)
        print(f"\nValidation Loss: {l_val}")

        l_training = np.mean(np.array(loss_train))
        print(f"Training Loss: {l_training}")
        val_targetv = np.concatenate(correct)

        if args.load_vicreg_path:
            torch.save(projector.state_dict(), f"{model_loc}/projector_{label}_last.pth")
        else:
            torch.save(gnn.state_dict(), f"{model_loc}/gnn_{label}_last.pth")
        if l_val < l_val_best:
            print("new best model")
            l_val_best = l_val
            if args.load_vicreg_path:
                torch.save(projector.state_dict(), f"{model_loc}/projector_{label}_best.pth")
                np.save(
                    f"{model_perf_loc}/projector_{label}_validation_target_vals.npy",
                    val_targetv,
                )
                np.save(
                    f"{model_perf_loc}/projector_{label}_validation_predicted_vals.npy",
                    predicted,
                )
                np.save(
                    f"{model_perf_loc}/projector_{label}_loss_train.npy",
                    np.array(loss_train),
                )
                np.save(
                    f"{model_perf_loc}/projector_{label}_loss_val.npy",
                    np.array(loss_val),
                )
            else:
                torch.save(gnn.state_dict(), f"{model_loc}/gnn_{label}_best.pth")
                np.save(
                    f"{model_perf_loc}/gnn_{label}_validation_target_vals.npy",
                    val_targetv,
                )
                np.save(
                    f"{model_perf_loc}/gnn_{label}_validation_predicted_vals.npy",
                    predicted,
                )
                np.save(
                    f"{model_perf_loc}/gnn_{label}_loss_train.npy",
                    np.array(loss_train),
                )
                np.save(
                    f"{model_perf_loc}/gnn_{label}_loss_val.npy",
                    np.array(loss_val),
                )

        acc_val = accuracy_score(val_targetv[:, 0], predicted[:, 0] > 0.5)
        print(f"Validation Accuracy: {acc_val}")


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune-mlp",
        default="256-256-2",
        help="Size and number of layers of the MLP finetuning head",
    )
    parser.add_argument(
        "--mlp",
        default="256-256-256",
        help="Size and number of layers of the MLP expander head",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        action="store",
        dest="train_path",
        default=f"{project_dir}/data/processed/train/",
        help="Input directory for training files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        action="store",
        dest="outdir",
        default=f"{project_dir}/models/",
        help="Output directory",
    )
    parser.add_argument(
        "--just-svs",
        action="store_true",
        default=False,
        help="Consider just secondary vertices in model",
    )
    parser.add_argument(
        "--just-tracks",
        action="store_true",
        default=False,
        help="Consider just tracks in model",
    )
    parser.add_argument("--De", type=int, action="store", dest="De", default=32, help="De")
    parser.add_argument("--Do", type=int, action="store", dest="Do", default=64, help="Do")
    parser.add_argument("--hidden", type=int, action="store", dest="hidden", default=128, help="hidden")
    parser.add_argument("--epoch", type=int, action="store", dest="epoch", default=100, help="Epochs")
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="new",
        help="a label for the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        dest="batch_size",
        default=1024,
        help="batch_size",
    )
    parser.add_argument(
        "--device",
        action="store",
        dest="device",
        default="cuda",
        help="device to train gnn; follow pytorch convention",
    )
    parser.add_argument(
        "--load-vicreg-path",
        type=str,
        action="store",
        default=None,
        help="Load weights from vicreg model if enabled",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="share parameters of backbone",
    )
    parser.add_argument(
        "--transform-inputs",
        type=int,
        action="store",
        dest="transform_inputs",
        default=64,
        help="transform_inputs",
    )

    args = parser.parse_args()
    main(args)
