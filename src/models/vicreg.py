import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import yaml
from torch import nn

from src.data.h5data import H5Data
from src.models.models import GraphNetEmbedding

if torch.cuda.is_available():
    import setGPU  # noqa: F401

project_dir = Path(__file__).resolve().parents[2]

train_path = "/ssl-jet-vol/hbb_interaction_network/data/processed/train/"
definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

N = defn["nobj_2"]  # number of charged particles
N_sv = defn["nobj_3"]  # number of SVs
n_targets = len(defn["reduced_labels"])  # number of classes
params = defn["features_2"]
params_sv = defn["features_3"]


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.x_transform = nn.Sequential(
            nn.BatchNorm1d(args.x_inputs),
            nn.Linear(args.x_inputs, args.transform_inputs),
            nn.BatchNorm1d(args.transform_inputs),
            nn.ReLU(),
        )
        self.y_transform = nn.Sequential(
            nn.BatchNorm1d(args.y_inputs),
            nn.Linear(args.y_inputs, args.transform_inputs),
            nn.BatchNorm1d(args.transform_inputs),
            nn.ReLU(),
        )
        self.x_backbone = args.x_backbone
        self.y_backbone = args.y_backbone
        self.N_x = self.x_backbone.N
        self.N_y = self.y_backbone.N
        self.embedding = args.Do
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        # x: [batch, x_inputs, N_x]
        # y: (batch, y_inputs, N_y]
        x = x.transpose(-1, -2).contiguous()  # [batch, N_x, x_inputs]
        y = y.transpose(-1, -2).contiguous()  # [batch, N_y, y_inputs]
        x = self.x_transform(x.view(-1, self.args.x_inputs)).view(
            -1, self.N_x, self.args.transform_inputs
        )  # [batch, N_x, transform_inputs]
        y = self.y_transform(y.view(-1, self.args.y_inputs)).view(
            -1, self.N_y, self.args.transform_inputs
        )  # [batch, N_y, transform_inputs]
        x = x.transpose(-1, -2).contiguous()  # [batch, x_inputs, N_x]
        y = y.transpose(-1, -2).contiguous()  # [batch, y_inputs, N_y]
        x = self.projector(self.x_backbone(x))
        y = self.projector(self.y_backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(
            self.num_features
        )

        loss = self.args.sim_coeff * repr_loss + self.args.std_coeff * std_loss + self.args.cov_coeff * cov_loss
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def main(args):

    files = glob.glob(os.path.join(train_path, "newdata_*.h5"))
    # take first 10% of files for validation
    # n_files_val should be 5 for full dataset
    n_files_val = max(1, int(0.1 * len(files)))
    files_val = files[:n_files_val]
    files_train = files[n_files_val:]

    n_epochs = args.epoch
    batch_size = args.batch_size

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

    n_train = data_train.count_data()
    n_val = data_val.count_data()

    args.x_inputs = len(params)
    args.y_inputs = len(params_sv)

    fr = nn.Sequential(
        nn.Linear(2 * args.transform_inputs, args.hidden),
        nn.BatchNorm1d(args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
        nn.BatchNorm1d(args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.De),
        nn.BatchNorm1d(args.De),
        nn.ReLU(),
    )
    fo = nn.Sequential(
        nn.Linear(args.transform_inputs + args.De, args.hidden),
        nn.BatchNorm1d(args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
        nn.BatchNorm1d(args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.Do),
        nn.BatchNorm1d(args.Do),
        nn.ReLU(),
    )
    args.x_backbone = GraphNetEmbedding(
        n_constituents=N,
        n_features=args.transform_inputs,
        fr=fr,
        fo=fo,
        De=args.De,
        Do=args.Do,
        device=args.device,
    )
    args.y_backbone = GraphNetEmbedding(
        n_constituents=N_sv,
        n_features=args.transform_inputs,
        fr=fr,
        fo=fo,
        De=args.De,
        Do=args.Do,
        device=args.device,
    )

    model = VICReg(args).to(args.device)

    train_its = int(n_train / batch_size)
    val_its = int(n_val / batch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_val = []
    loss_train = []
    l_val_best = 999999
    for m in range(n_epochs):
        print(f"Epoch {m}\n")
        loss_val_epoch = []
        train_iterator = data_train.generate_data()
        model.train()
        pbar = tqdm.tqdm(train_iterator, total=train_its)
        for element in pbar:
            (sub_X, _, _) = element
            x = torch.tensor(sub_X[2], dtype=torch.float, device=args.device)
            y = torch.tensor(sub_X[3], dtype=torch.float, device=args.device)
            optimizer.zero_grad()
            loss = model.forward(x, y)
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())
            pbar.set_description(f"Training loss: {loss.item():.4f}")
        model.eval()
        val_iterator = data_val.generate_data()
        pbar = tqdm.tqdm(val_iterator, total=val_its)
        for element in pbar:
            (sub_X, _, _) = element
            x = torch.tensor(sub_X[2], dtype=torch.float, device=args.device)
            y = torch.tensor(sub_X[3], dtype=torch.float, device=args.device)
            loss = model.forward(x, y)
            loss_val.append(loss.item())
            loss_val_epoch.append(loss.item())
            pbar.set_description(f"Validation loss: {loss.item():.4f}")

        l_val = np.mean(np.array(loss_val_epoch))
        if l_val < l_val_best:
            print("New best model")
            l_val_best = l_val
            torch.save(model.state_dict(), "vicreg_best.pth")
        torch.save(model.state_dict(), "vicreg_last.pth")
        np.save(
            "vicreg_loss_train.npy",
            np.array(loss_train),
        )
        np.save(
            "vicreg_loss_val.npy",
            np.array(loss_val),
        )


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        action="store",
        dest="outdir",
        default=f"{project_dir}/models/",
        help="Output directory",
    )
    parser.add_argument(
        "--transform-inputs",
        type=int,
        action="store",
        dest="transform_inputs",
        default=64,
        help="transform_inputs",
    )
    parser.add_argument("--De", type=int, action="store", dest="De", default=20, help="De")
    parser.add_argument("--Do", type=int, action="store", dest="Do", default=24, help="Do")
    parser.add_argument("--hidden", type=int, action="store", dest="hidden", default=60, help="hidden")
    parser.add_argument("--epoch", type=int, action="store", dest="epoch", default=100, help="Epochs")
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
        "--mlp",
        default="512-512-512",
        help="Size and number of layers of the MLP expander head",
    )
    parser.add_argument(
        "--sim-coeff",
        type=float,
        default=25.0,
        help="Invariance regularization loss coefficient",
    )
    parser.add_argument(
        "--std-coeff",
        type=float,
        default=25.0,
        help="Variance regularization loss coefficient",
    )
    parser.add_argument(
        "--cov-coeff",
        type=float,
        default=1.0,
        help="Covariance regularization loss coefficient",
    ) 

    args = parser.parse_args()
    main(args)
