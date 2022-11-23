import argparse
import glob
from pathlib import Path

import numpy as np
import torch

if torch.cuda.is_available():
    import setGPU  # noqa: F401

import tqdm
import yaml
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

from models import GraphNet, GraphNetSingle

project_dir = Path(__file__).resolve().parents[2]
definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

N = defn["nobj_2"]  # number of charged particles
N_sv = defn["nobj_3"]  # number of SVs
n_targets = len(defn["reduced_labels"])  # number of classes
spectators = defn["spectators"]
params = defn["features_2"]
params_sv = defn["features_3"]


def main(args, evaluating_test=True):  # noqa: C901

    device = args.device

    test_2 = []
    test_3 = []
    test_specs = []
    target_tests = []

    if evaluating_test:
        dataset = "test"
    else:
        dataset = "train"

    for test_file in sorted(glob.glob(f"{args.save_path}/{dataset}_*_features_2.npy")):
        test_2.append(np.load(test_file))
    test = np.concatenate(test_2)

    for test_file in sorted(glob.glob(f"{args.save_path}/{dataset}_*_features_3.npy")):
        test_3.append(np.load(test_file))
    test_sv = np.concatenate(test_3)

    for test_file in sorted(glob.glob(f"{args.save_path}/{dataset}_*_spectators.npy")):
        test_specs.append(np.load(test_file))
    test_spec = np.concatenate(test_specs)

    for test_file in sorted(glob.glob(f"{args.save_path}/{dataset}_*_truth.npy")):
        target_tests.append(np.load(test_file))
    target_test = np.concatenate(target_tests)

    fj_pt = test_spec[:, 0, 0]
    fj_eta = test_spec[:, 0, 1]
    fj_sdmass = test_spec[:, 0, 2]
    if args.no_undef:
        no_undef = np.sum(target_test, axis=1) == 1
    else:
        no_undef = fj_pt > -999  # no cut

    min_pt = args.min_pt  # 300
    max_pt = args.max_pt  # 2000
    min_eta = args.min_eta  # no cut
    max_eta = args.max_eta  # no cut
    min_msd = args.min_msd  # 40
    max_msd = args.max_msd  # 200

    for array in [test, test_sv, test_spec, target_test]:
        array = array[
            (fj_sdmass > min_msd)
            & (fj_sdmass < max_msd)
            & (fj_eta > min_eta)
            & (fj_eta < max_eta)
            & (fj_pt > min_pt)
            & (fj_pt < max_pt)
            & no_undef
        ]

    # Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    prediction = np.array([])

    batch_size = args.batch_size
    torch.cuda.empty_cache()

    if args.just_svs:
        gnn = GraphNetSingle(
            n_constituents=N_sv,
            n_targets=n_targets,
            params=len(params_sv),
            hidden=args.hidden,
            De=args.De,
            Do=args.Do,
            device=device,
        )
    elif args.just_tracks:
        gnn = GraphNetSingle(
            n_constituents=N,
            n_targets=n_targets,
            params=len(params),
            hidden=args.hidden,
            De=args.De,
            Do=args.Do,
            device=device,
        )
    else:
        gnn = GraphNet(
            n_constituents=N,
            n_targets=n_targets,
            params=len(params),
            hidden=args.hidden,
            n_vertices=N_sv,
            params_v=len(params_sv),
            De=args.De,
            Do=args.Do,
            device=device,
        )

    gnn.load_state_dict(torch.load(args.load_path))
    gnn.eval()
    print(sum(p.numel() for p in gnn.parameters() if p.requires_grad))

    for j in tqdm.tqdm(range(0, target_test.shape[0], batch_size)):
        dummy_input_1 = torch.from_numpy(test[j : j + batch_size]).to(device)
        dummy_input_2 = torch.from_numpy(test_sv[j : j + batch_size]).to(device)

        if args.just_svs:
            out_test = gnn(dummy_input_2)
        elif args.just_tracks:
            out_test = gnn(dummy_input_1)
        else:
            out_test = gnn(dummy_input_1, dummy_input_2)
        out_test = out_test.cpu().data.numpy()
        out_test = softmax(out_test, axis=1)
        if j == 0:
            prediction = out_test
        else:
            prediction = np.concatenate((prediction, out_test), axis=0)

    auc = roc_auc_score(target_test[:, 1], prediction[:, 1])
    print("AUC: ", auc)
    acc = accuracy_score(target_test[:, 0], prediction[:, 0] >= 0.5)
    print("Accuray: ", acc)
    # checking the sums
    target_sums = np.sum(target_test, 1)
    prediction_sums = np.sum(prediction, 1)
    idx = target_sums == 1
    print("Total: {}, Target: {}, Pred: {}".format(np.sum(idx), np.sum(target_sums[idx]), np.sum(prediction_sums[idx])))
    auc = roc_auc_score(target_test[idx][:, 1], prediction[idx][:, 1])
    print("AUC: ", auc)
    acc = accuracy_score(target_test[idx][:, 0], prediction[idx][:, 0] >= 0.5)
    print("Accuray 0: ", acc)
    acc = accuracy_score(target_test[idx][:, 1], prediction[idx][:, 1] >= 0.5)
    print("Accuray 1: ", acc)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument(
        "--save-path",
        type=str,
        action="store",
        default=f"{project_dir}/data/processed/test/",
        help="Input directory with testing files",
    )
    parser.add_argument(
        "--min-msd",
        type=float,
        action="store",
        default=-999.0,
        help="Min mSD for evaluation",
    )
    parser.add_argument(
        "--max-msd",
        type=float,
        action="store",
        default=9999.0,
        help="Max mSD for evaluation",
    )
    parser.add_argument(
        "--min-pt",
        type=float,
        action="store",
        default=-999.0,
        help="Min pT for evaluation",
    )
    parser.add_argument(
        "--max-pt",
        type=float,
        action="store",
        default=99999.0,
        help="Max pT for evaluation",
    )
    parser.add_argument(
        "--min-eta",
        type=float,
        action="store",
        default=-999.0,
        help="Min eta for evaluation",
    )
    parser.add_argument(
        "--max-eta",
        type=float,
        action="store",
        default=999.0,
        help="Max eta for evaluation",
    )
    parser.add_argument("--De", type=int, action="store", dest="De", default=20, help="De")
    parser.add_argument("--Do", type=int, action="store", dest="Do", default=24, help="Do")
    parser.add_argument("--hidden", type=int, action="store", dest="hidden", default=60, help="hidden")
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
        default="cpu",
        help="device to train gnn; follow pytorch convention",
    )
    parser.add_argument(
        "--no-undef",
        action="store_true",
        help="no undefined labels for evaluation",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        action="store",
        default=f"{project_dir}/models/trained_models/gnn_new_best.pth",
        help="Load weights from model if enabled",
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

    args = parser.parse_args()
    main(args, True)
