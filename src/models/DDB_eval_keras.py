import matplotlib as mpl
mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import glob
import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import pandas as pd
import sys
import imp
import os
import keras
from keras.layers import Input
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass    
import argparse

N = 60 # number of charged particles
N_sv = 5 # number of SVs 
n_targets = 2 # number of classes
if os.path.isdir('/storage/group/gpu/bigdata/BumbleB'):
    save_path = '/storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
elif os.path.isdir('/eos/user/w/woodson/IN'):
    save_path = '/eos/user/w/woodson/IN/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'

spectators = ['fj_pt',
              'fj_eta',
              'fj_sdmass',
              'fj_n_sdsubjets',
              'fj_doubleb',
              'fj_tau21',
              'fj_tau32',
              'npv',
              'npfcands',
              'ntracks',
              'nsv']

params_0 = ['fj_jetNTracks',
          'fj_nSV',
          'fj_tau0_trackEtaRel_0',
          'fj_tau0_trackEtaRel_1',
          'fj_tau0_trackEtaRel_2',
          'fj_tau1_trackEtaRel_0',
          'fj_tau1_trackEtaRel_1',
          'fj_tau1_trackEtaRel_2',
          'fj_tau_flightDistance2dSig_0',
          'fj_tau_flightDistance2dSig_1',
          'fj_tau_vertexDeltaR_0',
          'fj_tau_vertexEnergyRatio_0',
          'fj_tau_vertexEnergyRatio_1',
          'fj_tau_vertexMass_0',
          'fj_tau_vertexMass_1',
          'fj_trackSip2dSigAboveBottom_0',
          'fj_trackSip2dSigAboveBottom_1',
          'fj_trackSip2dSigAboveCharm_0',
          'fj_trackSipdSig_0',
          'fj_trackSipdSig_0_0',
          'fj_trackSipdSig_0_1',
          'fj_trackSipdSig_1',
          'fj_trackSipdSig_1_0',
          'fj_trackSipdSig_1_1',
          'fj_trackSipdSig_2',
          'fj_trackSipdSig_3',
          'fj_z_ratio'
          ]

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

def main(args):
    test_2_arrays = []
    test_3_arrays = []
    test_spec_arrays = []
    target_test_arrays = []
    
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
    target_test = np.concatenate(target_test_arrays)

    del test_2_arrays
    del test_3_arrays
    del test_spec_arrays
    del target_test_arrays
    test_2 = np.swapaxes(test_2, 1, 2)
    test_3 = np.swapaxes(test_3, 1, 2)
    test_spec = np.swapaxes(test_spec, 1, 2)
    print(test_2.shape)
    print(test_3.shape)
    print(target_test.shape)
    print(test_spec.shape)
    print(target_test.shape)
    fj_pt = test_spec[:,0,0]
    fj_eta = test_spec[:,1,0]
    fj_sdmass = test_spec[:,2,0]
    #no_undef = np.sum(target_test,axis=1) == 1
    no_undef = fj_pt > -999 # no cut

    min_pt = -999 #300
    max_pt = 99999 #2000
    min_eta = -999 # no cut
    max_eta = 999 # no cut
    min_msd = -999 #40
    max_msd = 9999 #200
    
    test_2 = test_2 [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef]
    test_3 = test_3 [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef ]
    test_spec = test_spec [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef ]
    target_test = target_test [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef  ]

    test_2 = np.swapaxes(test_2, 1, 2)
    test_3 = np.swapaxes(test_3, 1, 2)
    test_spec = np.swapaxes(test_spec, 1, 2)
    print(test_2.shape)
    print(test_3.shape)
    print(target_test.shape)
    print(test_spec.shape)
    print(target_test.shape)

    #Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    test = test_2
    params = params_2
    test_sv = test_3
    params_sv = params_3
    batch_size = 1024

    outdir = args.outdir
    label = 'new'

    prediction = np.array([])
    from ddb import model_DeepDoubleXReference
    keras_model = model_DeepDoubleXReference(inputs = [Input(shape=(N,len(params))),Input(shape=(N_sv,len(params_sv)))],
                                             num_classes = n_targets, scale_hidden = 2,
                                             hlf_input = None, datasets = ['cpf', 'sv'])

    keras_model.load_weights('%s/keras_model_best.h5'%outdir)

    keras_model.summary()

    for j in tqdm.tqdm(range(0, target_test.shape[0], batch_size)):
        out_test = keras_model.predict([test[j:j+batch_size],test_sv[j:j+batch_size]])
        if j==0:
            prediction = out_test
        else:
            prediction = np.concatenate((prediction, out_test),axis=0)        
        del out_test

    np.save('%s/truth_%s.npy'%(outdir,label),target_test)
    np.save('%s/prediction_%s.npy'%(outdir,label),prediction)
    print(target_test)
    print(prediction)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")

    args = parser.parse_args()
    main(args)
