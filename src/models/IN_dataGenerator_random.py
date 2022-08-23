from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import glob
import tqdm
import argparse
import sklearn.metrics as _m
import sklearn.model_selection

 
from models import GraphNet


#sys.path.insert(0, '/nfshome/jduarte/DL4Jets/mpi_learn/mpi_learn/train')
print(torch.__version__)

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
test_path ='//grand/RAPINS/ruike/new_hbb/test/'# '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
train_path = '//grand/RAPINS/ruike/new_hbb/train/'#'/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/'
NBINS = 40 # number of bins for loss function
MMAX = 200. # max value
MMIN = 40. # min value

N = 60 # number of charged particles
N_neu = 100 # number of neutral particles
N_sv = 5 # number of SVs 
n_targets = 2 # number of classes

params_1 = ['pfcand_ptrel',
          'pfcand_erel',
          'pfcand_phirel',
          'pfcand_etarel',
          'pfcand_deltaR',
          'pfcand_puppiw',
          'pfcand_drminsv',
          'pfcand_drsubjet1',
          'pfcand_drsubjet2',
          'pfcand_hcalFrac'
         ]

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



'''
#Deep double-b features 
params_2 = params_2[22:]
params_3 = params_2[11:13]
'''

def main(args):
    """ Main entry point of the app """
    
    #Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    params_neu = params_1
    params = params_2
    params_sv = params_3
    
    label = 'new'
    outdir = args.outdir
    vv_branch = args.vv_branch
    sv_branch = args.sv_branch
    os.system('mkdir -p %s'%outdir)

    
    if sv_branch: 
        gnn = GraphNet(N, n_targets, len(params), args.hidden, N_sv, len(params_sv),
                   vv_branch=int(vv_branch),
                   De=args.De,
                   Do=args.Do)
    else: 
        print("this is loading GraphNetnoSV")
    
    #Architecture with all-particles
    #gnn = GraphNetAllParticle(N, N_neu, n_targets, len(params), len(params_neu), args.hidden, N_sv, len(params_sv),vv_branch=int(vv_branch), De=args.De, Do=args.Do) 
    
    # pre load best model
    #gnn.load_state_dict(torch.load('out/gnn_new_best.pth'))

    n_epochs = 2# 200
    
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
    
    
    t_X1_tr = np.load("//grand/RAPINS/ruike/npy_data1/data_X1_tr.npy")
    t_X2_tr = np.load("//grand/RAPINS/ruike/npy_data1/data_X2_tr.npy")
    t_X3_tr = np.load("//grand/RAPINS/ruike/npy_data1/data_X3_tr.npy")
    
    t_X4_tr = np.load("//grand/RAPINS/ruike/npy_data1/data_X4_tr.npy")
    t_Y_tr = np.load("//grand/RAPINS/ruike/npy_data1/data_Y_tr.npy")
    t_Z_tr = np.load("//grand/RAPINS/ruike/npy_data1/data_Z_tr.npy")
    
    t_X1_te = np.load("//grand/RAPINS/ruike/npy_data1/data_X1_te.npy")
    t_X2_te = np.load("//grand/RAPINS/ruike/npy_data1/data_X2_te.npy")
    t_X3_te = np.load("//grand/RAPINS/ruike/npy_data1/data_X3_te.npy")
    
    t_X4_te = np.load("//grand/RAPINS/ruike/npy_data1/data_X4_te.npy")
    t_Y_te = np.load("//grand/RAPINS/ruike/npy_data1/data_Y_te.npy")
    t_Z_te = np.load("//grand/RAPINS/ruike/npy_data1/data_Z_te.npy")
    
    print(np.shape( t_X1_tr))
    print(np.shape( t_X2_tr))
    print(np.shape( t_X3_tr))
    print(np.shape( t_X4_tr))
    print(np.shape( t_Y_tr))
    print(np.shape( t_Z_tr))
      

    # print("seperate converting finished")
    t_X_tr =[t_X1_tr, t_X2_tr, t_X3_tr, t_X4_tr]
    t_Y_tr=[t_Y_tr]
    t_Z_tr =[t_Z_tr]
    print("mid for train finish numpy convert")
    t_X_te =[t_X1_te, t_X2_te, t_X3_te, t_X4_te]
    t_Y_te=[t_Y_te]
    t_Z_te =[t_Z_te]

    del t_X1_tr, t_X2_tr, t_X3_tr, t_X4_tr
    del t_X1_te, t_X2_te, t_X3_te, t_X4_te
    
    print('byte size for t_X1_tr', np.shape(t_X_tr[0]))#t_X_tr[0].itemsize* 
    print('byte size for t_X2_tr', np.shape(t_X_tr[1]))#t_X_tr[1].itemsize* 
    print('byte size for t_X3_tr',np.shape( t_X_tr[2]))#t_X_tr[2].itemsize* 
    print('byte size for t_X4_tr', np.shape(t_X_tr[3]))#t_X_tr[3].itemsize* 
    print('byte size for t_Y_tr', np.shape(t_Y_tr[0]))#t_Y_tr[0].itemsize* 
    print('byte size for t_Z_tr',  np.shape(t_Z_tr[0]))#t_Z_tr[0].itemsize*

    print('byte size for t_X1_te', np.shape(t_X_te[0]))#t_X_tr[0].itemsize* 
    print('byte size for t_X2_te', np.shape(t_X_te[1]))#t_X_tr[1].itemsize* 
    print('byte size for t_X3_te',np.shape( t_X_te[2]))#t_X_tr[2].itemsize* 
    print('byte size for t_X4_te', np.shape(t_X_te[3]))#t_X_tr[3].itemsize* 
    print('byte size for t_Y_te', np.shape(t_Y_te[0]))#t_Y_tr[0].itemsize* 
    print('byte size for t_Z_te',  np.shape(t_Z_te[0]))#t_Z_tr[0].itemsize*


    print("!!!11",len(t_X_te),len(t_Y_te),len(t_Z_te))
  

    batch_size =128
    for m in range(n_epochs):
        print("Epoch %s\n" % m)
        #torch.cuda.empty_cache()
        final_epoch = m
        lst = []
        loss_val = []
        loss_training = []
        correct = []
        tic = time.perf_counter()
        
        ####train################
        
        batch_num_tr = int(len(t_X_tr[1])/batch_size)
        print("batch num, X_tr_1, batch_size: ",batch_num_tr ,len(t_X_tr[1]), batch_size)
        for idx_ in tqdm.tqdm(range(batch_num_tr),total=batch_num_tr):
            if idx_ == batch_num_tr -1:
                training = t_X_tr[2][idx_*batch_size:-1]
                training_sv = t_X_tr[3][idx_*batch_size:-1] #sub_X[3]
                target = t_Y_tr[0][idx_*batch_size:-1]
                spec = t_Z_tr[0][idx_*batch_size:-1]

            else:
                training = t_X_tr[2][idx_*batch_size:(idx_+1)*batch_size]  #sub_X[2]
                #training_neu = sub_X[1]
                training_sv = t_X_tr[3][idx_*batch_size:(idx_+1)*batch_size] #sub_X[3]
                target = t_Y_tr[0][idx_*batch_size:(idx_+1)*batch_size]
                spec = t_Z_tr[0][idx_*batch_size:(idx_+1)*batch_size]

            trainingv = (torch.FloatTensor(training)).cuda()
            #trainingv_neu = (torch.FloatTensor(training_neu)).cuda()
            trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
            targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()
            optimizer.zero_grad()
            #out = gnn(trainingv.cuda(), trainingv_sv.cuda())
            
            #Input training dataset 
            if sv_branch: 
                out = gnn(trainingv.cuda(), trainingv_sv.cuda())
            else: 
                out = gnn(trainingv.cuda())
                
            l = loss(out, targetv.cuda())
            loss_training.append(l.item())
            l.backward()
            optimizer.step()
            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del trainingv, trainingv_sv, targetv,training, training_sv, target,spec
        toc = time.perf_counter()
        print(f"Training done in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()

        ##########test###################
        batch_num = int(len(t_X_te[1])/batch_size)
        for idx_ in tqdm.tqdm(range(batch_num),total=batch_num): ###
            if idx_ == batch_num -1:
                training = t_X_te[2][idx_*batch_size:-1]
                training_sv = t_X_te[3][idx_*batch_size:-1] #sub_X[3]
                target = t_Y_te[0][idx_*batch_size:-1]
                spec = t_Z_te[0][idx_*batch_size:-1]

            else:
                training = t_X_te[2][idx_*batch_size:(idx_+1)*batch_size]  #sub_X[2]
                #training_neu = sub_X[1]
                training_sv = t_X_te[3][idx_*batch_size:(idx_+1)*batch_size] #sub_X[3]
                target = t_Y_te[0][idx_*batch_size:(idx_+1)*batch_size]
                spec = t_Z_te[0][idx_*batch_size:(idx_+1)*batch_size]

            trainingv = (torch.FloatTensor(training)).cuda()
            #trainingv_neu = (torch.FloatTensor(training_neu)).cuda()
            trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
            targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()
            
            #Input validation dataset 
            if sv_branch: 
                out = gnn(trainingv.cuda(), trainingv_sv.cuda())
            else: 
                out = gnn(trainingv.cuda())
                
            lst.append(softmax(out).cpu().data.numpy())
            l_val = loss(out, targetv.cuda())
            loss_val.append(l_val.item())
            
            targetv_cpu = targetv.cpu().data.numpy()
        
            correct.append(target)
            del trainingv, trainingv_sv, targetv,training, training_sv, target,spec
        toc = time.perf_counter()
        print(f"Evaluation done in {toc - tic:0.4f} seconds")
        l_val = np.mean(np.array(loss_val))
    
        predicted = np.concatenate(lst) #(torch.FloatTensor(np.concatenate(lst))).to(device)
        print('\nValidation Loss: ', l_val)

        l_training = np.mean(np.array(loss_training))
        print('Training Loss: ', l_training)
        val_targetv = np.concatenate(correct) #torch.FloatTensor(np.array(correct)).cuda()
        
        torch.save(gnn.state_dict(), '../../models/trained_models/random_gnn_%s_last.pth'%label)
        if l_val < l_val_best:
            print("new best model")
            l_val_best = l_val
            torch.save(gnn.state_dict(), '../../models/trained_models/random_gnn_%s_best.pth'%label)
            
        print(val_targetv.shape, predicted.shape)
#         print(val_targetv, predicted)
        #add roc plot
#         path_ = './rocplots/random/'
#         util.roc_plot(val_targetv[:,0],predicted[:,0],m,path_)
        print("AUC score:",_m.roc_auc_score(val_targetv[:,0], predicted[:,0]))

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
    np.save('%s/random_acc_vals_validation_%s.npy'%(outdir,label),acc_vals_validation)
    np.save('%s/random_loss_vals_training_%s.npy'%(outdir,label),loss_vals_training)
    np.save('%s/random_loss_vals_validation_%s.npy'%(outdir,label),loss_vals_validation)
    np.save('%s/random_loss_std_validation_%s.npy'%(outdir,label),loss_std_validation)
    np.save('%s/random_loss_std_training_%s.npy'%(outdir,label),loss_std_training)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("sv_branch", help="Required positional argument")
    parser.add_argument("vv_branch", help="Required positional argument")
    
    # Optional arguments
    parser.add_argument("--De", type=int, action='store', dest='De', default = 5, help="De")
    parser.add_argument("--Do", type=int, action='store', dest='Do', default = 6, help="Do")
    parser.add_argument("--hidden", type=int, action='store', dest='hidden', default = 15, help="hidden")

    args = parser.parse_args()
    main(args)
