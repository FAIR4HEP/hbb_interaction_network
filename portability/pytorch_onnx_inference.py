import numpy as np
import torch
# import setGPU
import argparse
import onnx
import onnxruntime as ort
import warnings
import os
import sys
import time
import sklearn.metrics as _m
from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

import sys 
sys.path.append("..") 
from src.models.models import GraphNet
import tqdm

import glob
sv_branch = 1
N = 60 # number of charged particles
N_sv = 5 # number of SVs 
n_targets = 2 # number of classes
save_path =  './temp_test/'#'./test_hbb/'#
print( ort.get_device()  )

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

print("load start")
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


torch_soft_res = []
onnx_soft_res=[]
torch_res= []
onnx_res=[]
pytorch_time=[]
onnx_time=[]
label_ =[]

sample_size = 1800#000
batch_size= 128 
model_path = "./onnx_model/5_10_gnn_%s.onnx"%batch_size

label_batch = label_all[1:1+batch_size]


pbar = tqdm.tqdm(range(int(sample_size/batch_size)-1))
for i in pbar:   
    start_idx = i*batch_size
    label_batch = label_all[1+start_idx:1+start_idx+batch_size]
    
    
    dummy_input_1 = torch.from_numpy(test[1+start_idx:1+start_idx+batch_size]).cuda()
    dummy_input_2 = torch.from_numpy(test_sv[1+start_idx:1+start_idx+batch_size]).cuda() 
    
    #use pytorch gnn to predict
    start_time = time.perf_counter() 
    out_test = gnn(dummy_input_1, dummy_input_2)
    end_time = time.perf_counter() 
    temp_=end_time-start_time
    
    pytorch_time.append(temp_)
    
       
    # Load the ONNX model
    dummy_input_1 = test[1+start_idx:1+start_idx+batch_size]
    dummy_input_2 = test_sv[1+start_idx:1+start_idx+batch_size]
    model = onnx.load(model_path)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    ####print(onnx.helper.printable_graph(model.graph))

    
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    ort_session = ort.InferenceSession(model_path, options, providers=[('CUDAExecutionProvider')])

    # compute ONNX Runtime output prediction
    start_time = time.perf_counter()
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_1,
              ort_session.get_inputs()[1].name: dummy_input_2}

     
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.perf_counter() 
    #print(f"ONNXRuntime Inference in {toc - tic:0.4f} seconds")
    time_ = end_time-start_time
    onnx_time.append(time_)


    temp_onnx_res = ort_outs[0].tolist()
    temp_pytorch_res = out_test.cpu().detach().numpy().tolist()
    
    for x in label_batch:
        label_.append(x.tolist())
        
    for x in temp_onnx_res:
        onnx_res.append(x)
        x_ = softmax(x, axis=0)
        onnx_soft_res.append(x_.tolist())
    for x in temp_pytorch_res:
        torch_res.append(x)
        x_ = softmax(x, axis=0)
        torch_soft_res.append(x_.tolist())


clip_onnx_soft_res = []
clip_torch_soft_res=[]
clip_label = []
for i in range(len(label_)):
    if [0.,0,] == label_[i]:
        continue
    else:
        clip_onnx_soft_res.append(onnx_soft_res[i])
        clip_torch_soft_res.append(torch_soft_res[i])
        clip_label.append(label_[i])
        

# print(len(torch_res))

fpr_o, tpr_o, thresholds_p = _m.roc_curve(np.array(clip_label)[:,1], np.array(clip_onnx_soft_res)[:,1])
print("onnx acc",accuracy_score(np.array(clip_label)[:,1], np.array(clip_onnx_soft_res)[:,1]>0.5))
print("onnx auc",_m.auc(fpr_o, tpr_o) ) 
print("onnx",np.mean(onnx_time[1:]))

print("#############################")
fpr_p, tpr_p, thresholds_p = _m.roc_curve(np.array(clip_label)[:,1], np.array(clip_torch_soft_res)[:,1])
print("torch acc",accuracy_score(np.array(clip_label)[:,1], np.array(clip_torch_soft_res)[:,1]>0.5))
print("torch auc",_m.auc(fpr_p, tpr_p) ) 
print("pytorch",np.mean(pytorch_time[1:]))