import argparse
import glob
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import setGPU  # noqa: F401
import torch
import tqdm
import yaml
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

project_dir = Path(__file__).resolve().parents[2]
save_path = f"{project_dir}/data/processed/test/"
definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

N = defn["nobj_2"]  # number of charged particles
N_sv = defn["nobj_3"]  # number of SVs
n_targets = len(defn["reduced_labels"])  # number of classes
spectators = defn["spectators"]
params_2 = defn["features_2"]
params_3 = defn["features_3"]


def main(args, save_path="", evaluating_test=True):  # noqa: C901

    test_1_arrays = []
    test_2_arrays = []
    test_3_arrays = []
    test_spec_arrays = []
    target_test_arrays = []

    if evaluating_test:

        for test_file in sorted(glob.glob(save_path + "test_0_features_1.npy")):
            test_1_arrays.append(np.load(test_file))
        test_1 = np.concatenate(test_1_arrays)

        for test_file in sorted(glob.glob(save_path + "test_0_features_2.npy")):
            test_2_arrays.append(np.load(test_file))
        test_2 = np.concatenate(test_2_arrays)

        for test_file in sorted(glob.glob(save_path + "test_0_features_3.npy")):
            test_3_arrays.append(np.load(test_file))
        test_3 = np.concatenate(test_3_arrays)

        for test_file in sorted(glob.glob(save_path + "test_0_spectators_0.npy")):
            test_spec_arrays.append(np.load(test_file))
        test_spec = np.concatenate(test_spec_arrays)

        for test_file in sorted(glob.glob(save_path + "test_0_truth_0.npy")):
            target_test_arrays.append(np.load(test_file))
        target_test = np.concatenate(target_test_arrays)

    else:
        for test_file in sorted(glob.glob(save_path + "train_val_*_features_1.npy")):
            test_1_arrays.append(np.load(test_file))
        test_1 = np.concatenate(test_1_arrays)

        for test_file in sorted(glob.glob(save_path + "train_val_*_features_2.npy")):
            test_2_arrays.append(np.load(test_file))
        test_2 = np.concatenate(test_2_arrays)

        for test_file in sorted(glob.glob(save_path + "train_val_*_features_3.npy")):
            test_3_arrays.append(np.load(test_file))
        test_3 = np.concatenate(test_3_arrays)

        for test_file in sorted(glob.glob(save_path + "train_val_*_spectators_0.npy")):
            test_spec_arrays.append(np.load(test_file))
        test_spec = np.concatenate(test_spec_arrays)

        for test_file in sorted(glob.glob(save_path + "train_val_*_truth_0.npy")):
            target_test_arrays.append(np.load(test_file))
        target_test = np.concatenate(target_test_arrays)

    del test_1_arrays
    del test_2_arrays
    del test_3_arrays
    del test_spec_arrays
    del target_test_arrays
    test_1 = np.swapaxes(test_1, 1, 2)
    test_2 = np.swapaxes(test_2, 1, 2)
    test_3 = np.swapaxes(test_3, 1, 2)
    test_spec = np.swapaxes(test_spec, 1, 2)
    print(test_2.shape)
    print(test_3.shape)
    print(target_test.shape)
    print(test_spec.shape)
    print(target_test.shape)
    fj_pt = test_spec[:, 0, 0]
    fj_eta = test_spec[:, 1, 0]
    fj_sdmass = test_spec[:, 2, 0]
    # no_undef = np.sum(target_test,axis=1) == 1
    no_undef = fj_pt > -999  # no cut

    min_pt = -999  # 300
    max_pt = 99999  # 2000
    min_eta = -999  # no cut
    max_eta = 999  # no cut
    min_msd = -999  # 40
    max_msd = 9999  # 200

    test_1 = test_1[
        (fj_sdmass > min_msd)
        & (fj_sdmass < max_msd)
        & (fj_eta > min_eta)
        & (fj_eta < max_eta)
        & (fj_pt > min_pt)
        & (fj_pt < max_pt)
        & no_undef
    ]
    test_2 = test_2[
        (fj_sdmass > min_msd)
        & (fj_sdmass < max_msd)
        & (fj_eta > min_eta)
        & (fj_eta < max_eta)
        & (fj_pt > min_pt)
        & (fj_pt < max_pt)
        & no_undef
    ]
    test_3 = test_3[
        (fj_sdmass > min_msd)
        & (fj_sdmass < max_msd)
        & (fj_eta > min_eta)
        & (fj_eta < max_eta)
        & (fj_pt > min_pt)
        & (fj_pt < max_pt)
        & no_undef
    ]
    test_spec = test_spec[
        (fj_sdmass > min_msd)
        & (fj_sdmass < max_msd)
        & (fj_eta > min_eta)
        & (fj_eta < max_eta)
        & (fj_pt > min_pt)
        & (fj_pt < max_pt)
        & no_undef
    ]
    target_test = target_test[
        (fj_sdmass > min_msd)
        & (fj_sdmass < max_msd)
        & (fj_eta > min_eta)
        & (fj_eta < max_eta)
        & (fj_pt > min_pt)
        & (fj_pt < max_pt)
        & no_undef
    ]

    # Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    test = test_2
    test_sv = test_3
    params = params_2
    params_sv = params_3

    vv_branch = args.vv_branch
    set_onnx = args.set_onnx

    prediction = np.array([])
    batch_size = 1000  # 1024
    torch.cuda.empty_cache()

    from models import GraphNet

    gnn = GraphNet(
        N,
        n_targets,
        len(params),
        args.hidden,
        N_sv,
        len(params_sv),
        vv_branch=int(vv_branch),
        De=args.De,
        Do=args.Do,
    )

    if set_onnx is False:
        gnn.load_state_dict(torch.load("../../models/trained_models/gnn_new_best.pth"))
        print(sum(p.numel() for p in gnn.parameters() if p.requires_grad))

        for j in tqdm.tqdm(range(0, target_test.shape[0], batch_size)):
            dummy_input_1 = torch.from_numpy(test[j : j + batch_size]).cuda()
            dummy_input_2 = torch.from_numpy(test_sv[j : j + batch_size]).cuda()

            out_test = gnn(dummy_input_1, dummy_input_2)
            out_test = out_test.cpu().data.numpy()
            out_test = softmax(out_test, axis=1)
            if j == 0:
                prediction = out_test
            else:
                prediction = np.concatenate((prediction, out_test), axis=0)
            del out_test

    else:
        model_path = "../../models/trained_models/onnx_model/gnn_%s.onnx" % batch_size
        onnx_soft_res = []
        for i in tqdm.tqdm(range(0, target_test.shape[0], batch_size)):
            dummy_input_1 = test[i : i + batch_size]
            dummy_input_2 = test_sv[i : i + batch_size]

            # Load the ONNX model
            model = onnx.load(model_path)

            # Check that the IR is well formed
            onnx.checker.check_model(model)

            # Print a human readable representation of the graph
            # print(onnx.helper.printable_graph(model.graph))

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            ort_session = ort.InferenceSession(model_path, options, providers=[("CUDAExecutionProvider")])

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_1, ort_session.get_inputs()[1].name: dummy_input_2}
            ort_outs = ort_session.run(None, ort_inputs)

            temp_onnx_res = ort_outs[0]

            for x in temp_onnx_res:
                x_ = softmax(x, axis=0)
                onnx_soft_res.append(x_.tolist())

        prediction = np.array(onnx_soft_res)

    print(target_test.shape, prediction.shape)
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
    idx_bar = target_sums != 1
    print(target_test[idx_bar][0:10, :])


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("vv_branch", help="Required positional argument")
    # Optional arguments
    parser.add_argument("--De", type=int, action="store", dest="De", default=5, help="De")
    parser.add_argument("--Do", type=int, action="store", dest="Do", default=6, help="Do")
    parser.add_argument("--hidden", type=int, action="store", dest="hidden", default=15, help="hidden")
    parser.add_argument("--set_onnx", action="store_true", dest="set_onnx", default=False, help="set_onnx")

    args = parser.parse_args()
    main(args, save_path, True)
