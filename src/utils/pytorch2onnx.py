import glob
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

sys.path.append("..")
from models.models import GraphNet  # noqa: E402

sv_branch = 1

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

print(ort.get_device())


test_2_arrays = []
test_3_arrays = []
target_test_arrays = []
test_spec_arrays = []

for test_file in sorted(glob.glob(save_path + "test_*_features_2.npy")):
    test_2_arrays.append(np.load(test_file))
test_2 = np.concatenate(test_2_arrays)
for test_file in sorted(glob.glob(save_path + "test_*_features_3.npy")):
    test_3_arrays.append(np.load(test_file))
test_3 = np.concatenate(test_3_arrays)
for test_file in sorted(glob.glob(save_path + "test_*_spectators_0.npy")):
    test_spec_arrays.append(np.load(test_file))
test_spec = np.concatenate(test_spec_arrays)
for test_file in sorted(glob.glob(save_path + "test_*_truth_0.npy")):
    target_test_arrays.append(np.load(test_file))
label_all = np.concatenate(target_test_arrays)

print(len(label_all))

test = np.swapaxes(test_2, 1, 2)
test_sv = np.swapaxes(test_3, 1, 2)
test_spec = np.swapaxes(test_spec, 1, 2)
params = params_2
params_sv = params_3
label = "new"

gnn = GraphNet(
    N,
    n_targets,
    len(params),
    60,
    N_sv,
    len(params_sv),
    vv_branch=0,  # int(args.vv_branch),
    De=20,  # args.De,
    Do=24,  # args.Do,
    softmax=True,
)

gnn.load_state_dict(torch.load("../../models/trained_models/gnn_%s_last.pth" % (label), map_location=torch.device("cuda")))
torch.save(gnn.state_dict(), "../../models/trained_models/gnn_%s_last.pth" % (label))

torch_soft_res = []
onnx_soft_res = []
torch_res = []
onnx_res = []
pytorch_time = []
onnx_time = []
label_ = []

sample_size = 1800000
batch_sizes = [
    1000
]  # [200, 400, 600, 800, 1000, 1200, 1400, 1500, 1600, 1800, 2000, 2200, 2400, 2600, 3000, 3400, 3800, 4200]

for batch_size in batch_sizes:
    model_path = "../../models/trained_models/onnx_model/gnn_%s.onnx" % batch_size
    # build onnx model
    label_batch = label_all[1 : 1 + batch_size]
    dummy_input_1 = torch.from_numpy(test[1 : 1 + batch_size]).cuda()
    dummy_input_2 = torch.from_numpy(test_sv[1 : 1 + batch_size]).cuda()
    input_names = ["input_cpf", "input_sv"]
    output_names = ["output1"]

    torch.onnx.export(
        gnn,
        (dummy_input_1, dummy_input_2),
        model_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        dynamic_axes={
            input_names[0]: {0: "batch_size"},
            input_names[1]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        },
    )
