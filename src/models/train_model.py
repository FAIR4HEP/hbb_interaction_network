from __future__ import print_function

import argparse
import glob
import os, sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim

import setGPU
import tqdm
import argparse
import json
print(torch.__version__)

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
test_path = 'dataset/test/'
train_path = 'dataset/train/'
NBINS = 40 # number of bins for loss function
MMAX = 200. # max value
MMIN = 40. # min value

N = 60 # number of charged particles
N_sv = 5 # number of SVs 
n_targets = 2 # number of classes
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
    """ Main entry point of the app """
    
    model_dict = {}
    from data import h5data
    files = glob.glob(train_path + "/newdata_*.h5")
    files_val = files[:5] # take first 5 for validation
    files_train = files[5:] # take rest for training
    
    outdir = args.outdir
    vv_branch = args.vv_branch
    drop_rate = args.drop_rate
    load_def = args.load_def
    
    if args.drop_pfeatures != '':
        drop_pfeatures = list(map(int, str(args.drop_pfeatures).split(',')))
    else:
        drop_pfeatures = []

    if args.drop_svfeatures != '':
        drop_svfeatures = list(map(int, str(args.drop_svfeatures).split(',')))
    else:
        drop_svfeatures = []
    
    label = args.label
    if label == '' and drop_rate != 0:
        label = 'new_DR' + str(int(drop_rate*100.0))
    elif label == '' and drop_rate == 0:
        label = 'new'
    if len(drop_pfeatures) > 0:
        print("The following particle candidate features to be dropped: ", drop_pfeatures)
    if len(drop_svfeatures) > 0:
        print("The following secondary vertex features to be dropped: ", drop_svfeatures)
    n_epochs = args.epoch
    batch_size = args.batch_size
    model_loc = "{}/trained_models/".format(outdir)
    model_perf_loc = "{}/model_performances".format(outdir)
    model_dict_loc = "{}/model_dicts".format(outdir)
    os.system('mkdir -p {} {} {}'.format(model_loc, model_perf_loc, model_dict_loc))

    ## Saving the model's metadata as a json dict
    for arg in vars(args):
        model_dict[arg] = getattr(args, arg)
    f_model = open("{}/gnn_{}_model_metadata.json".format(model_dict_loc, label), "w")
    json.dump(model_dict, f_model, indent=3)
    f_model.close()
    

    ## Get the training and validation data
    data_train = h5data.H5Data(batch_size = batch_size,
                               cache = None,
                               preloading=0,
                               features_name='training_subgroup', 
                               labels_name='target_subgroup',
                               spectators_name='spectator_subgroup')
    data_train.set_file_names(files_train)
    data_val = h5data.H5Data(batch_size = batch_size,
                             cache = None,
                             preloading=0,
                             features_name='training_subgroup', 
                             labels_name='target_subgroup',
                             spectators_name='spectator_subgroup')
    data_val.set_file_names(files_val)

    n_val=data_val.count_data()
    n_train=data_train.count_data()
    print("val data:", n_val)
    print("train data:", n_train)

    from models import GraphNet

    gnn = GraphNet(n_constituents = N,
                       n_targets = n_targets,
                       params = len(params) - len(drop_pfeatures),
                       hidden = args.hidden,
                       n_vertices = N_sv,
                       params_v = len(params_sv) - len(drop_svfeatures),
                       vv_branch = int(vv_branch),
                       De = args.De,
                       Do = args.Do)

    #    ### N = Number of charged particles (60)
    #    ### n_targets = 2 (number of target_class)
    #    ### hidden = number of nodes in hidden layers
    #    ### params = number of features for each charged particle (30)
    #    ### n_vertices = number of secondary vertices (5)
    #    ### params_v = number of features for secondary vertices (14)
    #    ### vv_branch = to allow vv_branch ? (0 or False by default)
    #    ### De = Output dimension of particle-particle interaction NN (fR)
    #    ### Do = Output dimension of pre-aggregator transformation NN (fO) 
    
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
            if key not in ['fr1_pv.weight', 'fr1.weight', 'fo1.weight']:
                if new_state_dict[key].shape != def_state_dict[key].shape:
                    print("Tensor shapes don't match for key='{}': old = ({},{}); new = ({},{}): not updating it".format(key,
                                                                                                                         def_state_dict[key].shape[0],
                                                                                                                         def_state_dict[key].shape[1],
                                                                                                                         new_state_dict[key].shape[0],
                                                                                                                         new_state_dict[key].shape[1]))
                else:
                    new_state_dict[key] = def_state_dict[key].clone()
            else:
                if key == 'fr1_pv.weight':
                    indices_to_keep = [i for i in range(len(params)) if i not in drop_pfeatures] + \
                                      [len(params) + i for i in range(len(params_sv)) if i not in drop_svfeatures]

                if key == 'fr1.weight':
                    indices_to_keep = [i for i in range(len(params)) if i not in drop_pfeatures] + \
                                      [len(params) + i for i in range(len(params)) if i not in drop_pfeatures]

                if key == 'fo1.weight':
                    indices_to_keep = [i for i in range(len(params)) if i not in drop_pfeatures] + \
                                      list(range(len(params), len(params)+2*args.De))
                    
                new_tensor = def_state_dict[key][:,indices_to_keep]
                
                if new_state_dict[key].shape != new_tensor.shape:
                    print("Tensor shapes don't match for key='{}': modified old = ({},{}); new = ({},{}): not updating it".format(key,
                                                                                                                                  new_tensor.shape[0],
                                                                                                                                  new_tensor.shape[1],
                                                                                                                                  new_state_dict[key].shape[0],
                                                                                                                                  new_state_dict[key].shape[1]))

                else:    
                    new_state_dict[key] = new_tensor.clone()
        gnn.load_state_dict(new_state_dict)
            
                                    
    
    loss = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(gnn.parameters(), lr = 0.0001)
    loss_vals_training = np.zeros(n_epochs)
    loss_std_training = np.zeros(n_epochs)
    loss_vals_validation = np.zeros(n_epochs)
    loss_std_validation = np.zeros(n_epochs)

    acc_vals_training = np.zeros(n_epochs)
    acc_vals_validation = np.zeros(n_epochs)
    acc_std_training = np.zeros(n_epochs)
    acc_std_validation = np.zeros(n_epochs)

    final_epoch = 0
    l_val_best = 99999
    
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
    softmax = torch.nn.Softmax(dim=1)
    import time
    
    for m in range(n_epochs):
        print("Epoch %s\n" % m)
        #torch.cuda.empty_cache()
        final_epoch = m
        lst = []
        loss_val = []
        loss_training = []
        correct = []
        tic = time.perf_counter()
        sig_count = 0
        data_dropped = 0
        for sub_X, sub_Y, _ in tqdm.tqdm(data_train.generate_data(), total=int(n_train / batch_size)):
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]
            trainingv = (torch.FloatTensor(training)).cuda()
            trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
            targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()
            if drop_rate > 0:
                keep_indices = (targetv==0)
                sig_count += batch_size - torch.sum(keep_indices).item()
                to_turn_off = int((batch_size - torch.sum(keep_indices).item())*drop_rate)
                psum = torch.cumsum(~keep_indices,0)
                to_keep_after = torch.sum(psum <= to_turn_off).item()
                keep_indices[to_keep_after:] = torch.Tensor([True]*(batch_size - to_keep_after))
                data_dropped += batch_size - torch.sum(keep_indices).item()
                trainingv = trainingv[keep_indices]
                trainingv_sv = trainingv_sv[keep_indices]
                targetv = targetv[keep_indices]
            if len(drop_pfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params), 1, dtype=int) if i not in drop_pfeatures]
                trainingv = trainingv[:,keep_features,:]
            if len(drop_svfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params_sv), 1, dtype=int) if i not in drop_svfeatures]
                trainingv_sv = trainingv_sv[:,keep_features,:]
            
            optimizer.zero_grad()
            out = gnn(trainingv.cuda(), trainingv_sv.cuda())
            l = loss(out, targetv.cuda())
            loss_training.append(l.item())
            l.backward()
            optimizer.step()
            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del trainingv, trainingv_sv, targetv

        if drop_rate > 0.:
            print("Signal Count: {}, Data Dropped: {}".format(sig_count, data_dropped))
        toc = time.perf_counter()
        print(f"Training done in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
     
        for sub_X, sub_Y, _ in tqdm.tqdm(data_val.generate_data(), total=n_val / batch_size):
            training = sub_X[2]
            training_sv = sub_X[3]
            target = sub_Y[0]
            trainingv = (torch.FloatTensor(training)).cuda()
            trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
            targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()

            if len(drop_pfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params), 1, dtype=int) if i not in drop_pfeatures]
                trainingv = trainingv[:,keep_features,:]
            if len(drop_svfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params_sv), 1, dtype=int) if i not in drop_svfeatures]
                trainingv_sv = trainingv_sv[:,keep_features,:]
            
            out = gnn(trainingv.cuda(), trainingv_sv.cuda())
                
            lst.append(softmax(out).cpu().data.numpy())
            l_val = loss(out, targetv.cuda())
            loss_val.append(l_val.item())
            
            targetv_cpu = targetv.cpu().data.numpy()
        
            correct.append(target)
            del trainingv, trainingv_sv, targetv
        toc = time.perf_counter()
        print(f"Evaluation done in {toc - tic:0.4f} seconds")
        l_val = np.mean(np.array(loss_val))
    
        predicted = np.concatenate(lst) 
        print('\nValidation Loss: ', l_val)

        l_training = np.mean(np.array(loss_training))
        print('Training Loss: ', l_training)
        val_targetv = np.concatenate(correct) 
        
        torch.save(gnn.state_dict(), '%s/gnn_%s_last.pth'%(model_loc,label))
        if l_val < l_val_best:
            print("new best model")
            l_val_best = l_val
            torch.save(gnn.state_dict(), '%s/gnn_%s_best.pth'%(model_loc,label))
            np.save('%s/validation_target_vals_%s.npy'%(model_perf_loc,label),val_targetv)
            np.save('%s/validation_predicted_vals_%s.npy'%(model_perf_loc,label),predicted)
            
        print(val_targetv.shape, predicted.shape)
        print(val_targetv, predicted)
        acc_vals_validation[m] = accuracy_score(val_targetv[:,0],predicted[:,0]>0.5)
        print("Validation Accuracy: ", acc_vals_validation[m])
        loss_vals_training[m] = l_training
        loss_vals_validation[m] = l_val
        loss_std_validation[m] = np.std(np.array(loss_val))
        loss_std_training[m] = np.std(np.array(loss_training))
        if m > 8 and all(loss_vals_validation[max(0, m - 8):m] > min(np.append(loss_vals_validation[0:max(0, m - 8)], 200))):
            print('Early Stopping...')
            print(loss_vals_training, '\n', np.diff(loss_vals_training))
            break
        print()

    acc_vals_validation = acc_vals_validation[:(final_epoch + 1)]
    loss_vals_training = loss_vals_training[:(final_epoch + 1)]
    loss_vals_validation = loss_vals_validation[:(final_epoch + 1)]
    loss_std_validation = loss_std_validation[:(final_epoch + 1)]
    loss_std_training = loss_std_training[:(final_epoch)]
    np.save('%s/acc_vals_validation_%s.npy'%(model_perf_loc,label),acc_vals_validation)
    np.save('%s/loss_vals_training_%s.npy'%(model_perf_loc,label),loss_vals_training)
    np.save('%s/loss_vals_validation_%s.npy'%(model_perf_loc,label),loss_vals_validation)
    np.save('%s/loss_std_validation_%s.npy'%(model_perf_loc,label),loss_std_validation)
    np.save('%s/loss_std_training_%s.npy'%(model_perf_loc,label),loss_std_training)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--outdir", type=str, action='store', dest='outdir', default='../../models/', help="Output directory")
    parser.add_argument("--vv_branch", action='store_true', dest='vv_branch', default=False, help="Consider vertex-vertex interaction in model")
    
    parser.add_argument("--De", type=int, action='store', dest='De', default = 20, help="De")
    parser.add_argument("--Do", type=int, action='store', dest='Do', default = 24, help="Do")
    parser.add_argument("--hidden", type=int, action='store', dest='hidden', default = 60, help="hidden")
    parser.add_argument("--drop-rate", type=float, action='store', dest='drop_rate', default = 0., help="Signal Drop rate")
    parser.add_argument("--epoch", type=int, action='store', dest='epoch', default = 100, help="Epochs")
    parser.add_argument("--drop-pfeatures", type=str, action='store', dest='drop_pfeatures', default = '', help="comma separated indices of the particle candidate features to be dropped")
    parser.add_argument("--drop-svfeatures", type=str, action='store', dest='drop_svfeatures', default = '', help="comma separated indices of the secondary vertex features to be dropped")
    parser.add_argument("--label", type=str, action='store', dest='label', default = '', help="a label for the model")
    parser.add_argument("--batch-size", type=int, action='store', dest='batch_size', default = 128, help="batch_size")
    parser.add_argument("--load-def", action='store_true', dest='load_def', default = False, help="Load weights from default model if enabled")
    
    args = parser.parse_args()
    main(args)
