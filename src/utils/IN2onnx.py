import torch
import matplotlib.pyplot as plt
import numpy as np
from gnn import GraphNetnoSV
from gnn import GraphNet
from gnn import GraphNetAllParticle
from data import H5Data
import glob
import json


params = ['track_ptrel',
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

params_sv = ['sv_ptrel',
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


batch_size = 1024
files = glob.glob("dataset/train/newdata_*.h5")
files_val = files[:1] # take first file for validation
data_val = H5Data(batch_size = batch_size,
                  cache = None,
                  preloading=0,
                  features_name='training_subgroup',
                  labels_name='target_subgroup',
                  spectators_name='spectator_subgroup')
data_val.set_file_names(files_val)
n_val = data_val.count_data()
print("Data count: ", n_val)

for sub_X,sub_Y,sub_Z in data_val.generate_data():
    dummy_input_1 = (torch.FloatTensor(sub_X[2])).cuda()
    dummy_input_2 = (torch.FloatTensor(sub_X[3])).cuda()
    break


softmax = torch.nn.Softmax(dim=1)
gnn_0 = GraphNet(60, 2, len(params), 60, 5, len(params_sv),
                 vv_branch=0,
                 De=20,
                 Do=24)
gnn_0.load_state_dict(torch.load('IN_training/gnn_new_DR0_best.pth'))



input_names = ['input_cpf', 'input_sv']
output_names = ['output1']
torch.onnx.export(gnn_0,
                  (dummy_input_1, dummy_input_2),
                  "IN_training/gnn_new_DR0_best.onnx",
                  verbose=True,
                  input_names = input_names,
                  output_names = output_names,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  dynamic_axes = {input_names[0]: {0: 'batch_size'}, 
                                  input_names[1]: {0: 'batch_size'}, 
                                  output_names[0]: {0: 'batch_size'}})

# def eval(model,
#          save_data = False):
#     lst = []
#     correct = []
    
#     with torch.no_grad():
#         for sub_X,sub_Y,sub_Z in data_val.generate_data():
#             training = sub_X[2]
#             training_sv = sub_X[3]
#             if save_data:
#                 training_all.append(training)
#                 training_sv_all.append(training_sv)
#             target = sub_Y[0]
#             spec = sub_Z[0]
#             trainingv = (torch.FloatTensor(training)).cuda()
#             trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
#             targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()
#             print(trainingv.size(), trainingv_sv.size())
#             out = model.forward(trainingv.cuda(), trainingv_sv.cuda())
#             print(out.size())
#             lst.append(softmax(out).cpu().data.numpy())
#             correct.append(target)
    
#     predicted = np.concatenate(lst)
#     val_targetv = np.concatenate(correct)
    
#     return predicted, val_targetv

# pred_0, target_0 = eval(gnn_0, save_data = True)
# print(pred_0)
