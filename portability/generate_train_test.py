from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import os
import numpy as np
import pandas as pd
# import setGPU
from utils.data import H5Data 
import glob
import sys
import tqdm
import argparse
import sklearn.metrics as _m
import sklearn.model_selection

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

def main():
    """ Main entry point of the app """
    
    #Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    params_neu = params_1
    params = params_2
    params_sv = params_3
    
    files = glob.glob(train_path + "/newdata_*.h5")
    # files_val = files[:5] # take first 5 for validation
    files_train = files # take rest for training
    
    label = 'new'

    batch_size =5229076 
    data_train = H5Data(batch_size = batch_size,
                        cache = None,
                        preloading=0,
                        features_name='training_subgroup', 
                        labels_name='target_subgroup',
                        spectators_name='spectator_subgroup')
    data_train.set_file_names(files_train)
#     data_val = H5Data(batch_size = batch_size,
#                       cache = None,
#                       preloading=0,
#                       features_name='training_subgroup', 
#                       labels_name='target_subgroup',
#                       spectators_name='spectator_subgroup')
#     data_val.set_file_names(files_val)

#     n_val=data_val.count_data()
    n_train=data_train.count_data()
    batch_size = n_train
#     print("val data:", n_val)
    print("train data:", n_train)


    import time
    
    t_X1 = []
    t_X2 = []
    t_X3 = []
    t_X4 = []
    t_Y = []
    t_Z = []
    # import time
    start_time = time.time()
    for sub_X,sub_Y,sub_Z in tqdm.tqdm(data_train.generate_data(),total=n_train/batch_size):
        #print("read_num:",n_train/batch_size)
        #print(np.shape(sub_X[0]))
        t_X1 = sub_X[0]
        t_X2 = sub_X[1]
        t_X3 = sub_X[2]
        t_X4 = sub_X[3]
        t_Y = sub_Y[0]
        t_Z = sub_Z[0]
    end_time = time.time()
    print("time for load data:",end_time - start_time)
    

    ### split using rand
    #index_l = list(range(len(t_Z)))
    #zipped = zip(t_X,t_Y,t_Z)
    print(len(t_Z))
    print("splitting test and train!")
    index_list = list(range(len(t_Z)))
    t_X1_tr,t_X1_te,t_X2_tr,t_X2_te,t_X3_tr,t_X3_te,t_X4_tr,t_X4_te,\
    t_Y_tr,t_Y_te,t_Z_tr,t_Z_te = sklearn.model_selection.train_test_split(t_X1,t_X2,t_X3,t_X4,t_Y,t_Z, test_size =0.09, train_size = 0.91)

    
    ind_tr,ind_val = sklearn.model_selection.train_test_split(index_list, test_size =0.09, train_size = 0.91)
    print("X1 start")
    t_X1_tr = t_X1[ind_tr]
    t_X1_te = t_X1[ind_val]
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X1_tr.npy",t_X1_tr)
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X1_te.npy",t_X1_te)
    del t_X1

    print("X2")
    t_X2_tr = t_X2[ind_tr]
    t_X2_te = t_X2[ind_val]
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X2_tr.npy",t_X2_tr)
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X2_te.npy",t_X2_te)
    del t_X2

    print("X3")
    t_X3_tr = t_X3[ind_tr]
    t_X3_te = t_X3[ind_val]
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X3_tr.npy",t_X3_tr)
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X3_te.npy",t_X3_te)
    del t_X3

    print("X4")
    t_X4_tr = t_X4[ind_tr]
    t_X4_te = t_X4[ind_val]
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X4_tr.npy",t_X4_tr)
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_X4_te.npy",t_X4_te)
    del t_X4

    print("Y")
    t_Y_tr = t_Y[ind_tr]
    t_Y_te = t_Y[ind_val]
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_Y_tr.npy",t_Y_tr)
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_Y_te.npy",t_Y_te)
    del t_Y

    print("Z")
    t_Z_tr = t_Z[ind_tr]
    t_Z_te = t_Z[ind_val]
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_Z_tr.npy",t_Z_tr)
    np.save("//grand/RAPINS/ruike/new_hbb/npy_data1/data_Z_te.npy",t_Z_te)
    del t_Z


    #del t_X1,t_X2,t_X3,t_X4,t_Y,t_Z


    print("splitting done")

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


    print("!!!11",len(t_X_te),np.shape(t_X_te[0]),np.shape(t_X_te[1]),np.shape(t_X_te[2]),np.shape(t_X_te[3]),len(t_Y_te),len(t_Z_te))
 

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
