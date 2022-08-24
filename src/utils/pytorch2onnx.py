import numpy as np
import torch
import onnx
import onnxruntime as ort
import sys
import glob
import sklearn.metrics as _m
import sys
sys.path.append("..")
from models.models import GraphNet


sv_branch = 1
N = 60 # number of charged particles
N_sv = 5 # number of SVs
n_targets = 2 # number of classes
save_path = '//grand/RAPINS/ruike/new_hbb/test/'

print( ort.get_device())

params_2 = ['track_ptrel',
          'track_erel',
          'track_phirel',
          'track_etarel',
          'track_deltaR',
          'track_drminsv',
          'track_drsubjet1',
          'track_drsubjet2',
          'track_dz',
          'track_dzsig',
          'track_dxy',
          'track_dxysig',
          'track_normchi2',
          'track_quality',
          'track_dptdpt',
          'track_detadeta',
          'track_dphidphi',
          'track_dxydxy',
          'track_dzdz',
          'track_dxydz',
          'track_dphidxy',
          'track_dlambdadz',
          'trackBTag_EtaRel',
          'trackBTag_PtRatio',
          'trackBTag_PParRatio',
          'trackBTag_Sip2dVal',
          'trackBTag_Sip2dSig',
          'trackBTag_Sip3dVal',
          'trackBTag_Sip3dSig',
          'trackBTag_JetDistVal'
         ]
params_3 = ['sv_ptrel',
          'sv_erel',
          'sv_phirel',
          'sv_etarel',
          'sv_deltaR',
          'sv_pt',
          'sv_mass',
          'sv_ntracks',
          'sv_normchi2',
          'sv_dxy',
          'sv_dxysig',
          'sv_d3d',
          'sv_d3dsig',
          'sv_costhetasvpv'
         ]


test_2_arrays = []
test_3_arrays = []
target_test_arrays = []
test_spec_arrays = []

for test_file in sorted(glob.glob(save_path + 'test_*_features_2.npy')):
    test_2_arrays.append(np.load(test_file))
test_2 = np.concatenate(test_2_arrays)
for test_file in sorted(glob.glob(save_path + 'test_*_features_3.npy')):
    test_3_arrays.append(np.load(test_file))
test_3 = np.concatenate(test_3_arrays)
for test_file in sorted(glob.glob(save_path + 'test_*_spectators_0.npy')):
    test_spec_arrays.append(np.load(test_file))
test_spec = np.concatenate(test_spec_arrays)
for test_file in sorted(glob.glob(save_path + 'test_*_truth_0.npy')):
    target_test_arrays.append(np.load(test_file))
label_all = np.concatenate(target_test_arrays)

print(len(label_all))

test_2 = np.swapaxes(test_2, 1, 2)
test_3 = np.swapaxes(test_3, 1, 2)
test_spec = np.swapaxes(test_spec, 1, 2)

fj_pt = test_spec[:,0,0]
fj_eta = test_spec[:,1,0]
fj_sdmass = test_spec[:,2,0]
print("before",test_2.shape)
print("before",test_3.shape)

no_undef = fj_pt > -999 # no cut
min_pt = -999 #300
max_pt = 99999 #2000
min_eta = -999 # no cut
max_eta = 999 # no cut
min_msd = -999 #40
max_msd = 9999 #200

test_2 = test_2[(fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef]
test_3 = test_3[(fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef]
test_spec = test_spec[(fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef ]
label_all = label_all[(fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef ]

print("after",test_2.shape)
print("after",test_3.shape)

test = test_2
params = params_2
test_sv = test_3
params_sv = params_3
label = 'new'

gnn = GraphNet(N, n_targets, len(params), 60, N_sv, len(params_sv),
               vv_branch=0, #int(args.vv_branch),
               De=20,#args.De,
               Do=24 ,#args.Do,
               softmax=True)

gnn.load_state_dict(torch.load('../../models/trained_models/random_gnn_%s_last.pth'%(label), map_location=torch.device('cuda')))
torch.save(gnn.state_dict(),'../../models/trained_models/random_gnn_%s_last.pth'%(label))

torch_soft_res = []
onnx_soft_res=[]
torch_res= []
onnx_res=[]
pytorch_time=[]
onnx_time=[]
label_ =[]

sample_size = 1800000
batch_sizes= [200,400,600,800,1000,1200,1400,1500,1600,1800,2000,2200,2400,2600,3000,3400,3800,4200]

for batch_size in batch_sizes:
    model_path = "../../models/trained_models/onnx_model/5_10_gnn_%s.onnx"%batch_size
    #build onnx model
    label_batch = label_all[1:1+batch_size]
    dummy_input_1 = torch.from_numpy(test[1:1+batch_size]).cuda()
    dummy_input_2 = torch.from_numpy(test_sv[1:1+batch_size]).cuda()
    input_names = ['input_cpf', 'input_sv']
    output_names = ['output1']

    torch.onnx.export(gnn, (dummy_input_1, dummy_input_2), model_path, verbose=True,
                    input_names = input_names, output_names = output_names,
                    export_params=True,    # store the trained parameter weights inside the model file
                    opset_version=11,      # the ONNX version to export the model to
                    dynamic_axes = {input_names[0]: {0: 'batch_size'},
                                input_names[1]: {0: 'batch_size'},
                                output_names[0]: {0: 'batch_size'}})
