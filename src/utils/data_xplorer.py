import os, sys, glob, tqdm
import torch
import numpy as np
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
#test_path = '/storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
train_path = 'dataset/train/'

from data import H5Data
files = glob.glob(train_path + "/newdata_*.h5")
files_val = files[:5] # take first 5 for validation
files_train = files[5:] # take rest for training

batch_size = 128
data_train = H5Data(batch_size = batch_size,
                    cache = None,
                    preloading=0,
                    features_name='training_subgroup', 
                    labels_name='target_subgroup',
                    spectators_name='spectator_subgroup')
data_train.set_file_names(files_train)

data_val = H5Data(batch_size = batch_size,
                    cache = None,
                    preloading=0,
                    features_name='training_subgroup', 
                    labels_name='target_subgroup',
                    spectators_name='spectator_subgroup')
data_val.set_file_names(files_val)

n_train=data_train.count_data()
n_val=data_val.count_data()
n0 = 0

for sub_X,sub_Y,sub_Z in tqdm.tqdm(data_train.generate_data(),total=n_train/batch_size):
    training = sub_X[2]
    #training_neu = sub_X[1]
    training_sv = sub_X[3]
    target = sub_Y[0]
    spec = sub_Z[0]
    targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long())
    keep_indices = (targetv==0)
    #print(keep_indices)
    # print("Initial Drop Count:", batch_size - torch.sum(keep_indices).item())
    n0 += torch.sum(keep_indices).item()
    # to_turn_on = torch.sum(keep_indices).item() + int(2*(batch_size - torch.sum(keep_indices).item())/3)
    # to_turn_off = int(1*(batch_size - torch.sum(keep_indices).item())/3)
    # print("Desired drop count: ", to_turn_off)
    # idx = 0
    # # while to_turn_on > 0:
    # #     #print(keep_indices[idx].item())
    # #     if not keep_indices[idx].item():
    # #         #print("Now change it")
    # #         to_turn_on -= 1
    # #         keep_indices[idx] = torch.tensor(True)
    # #     idx += 1
    # psum = torch.cumsum(~keep_indices,0)
    # print(psum)
    # print(psum[psum <= to_turn_off])
    # to_keep_after = torch.sum(psum <= to_turn_off).item()
    # print("To Keep After: ", to_keep_after)
    # keep_indices[to_keep_after:] = torch.Tensor([True]*(batch_size - to_keep_after))
    # print("Final Drop Count:", batch_size - torch.sum(keep_indices).item())
    # print(keep_indices)
    # break
   

print("Total data:", n_train)
print("target=0 data:", n0)

n0 = 0

for sub_X,sub_Y,sub_Z in tqdm.tqdm(data_val.generate_data(),total=n_val/batch_size):
    training = sub_X[2]
    #training_neu = sub_X[1]
    training_sv = sub_X[3]
    target = sub_Y[0]
    spec = sub_Z[0]
    targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long())
    keep_indices = (targetv==0)
    #print(keep_indices)
    # print("Initial Drop Count:", batch_size - torch.sum(keep_indices).item())
    n0 += torch.sum(keep_indices).item()
    # to_turn_on = torch.sum(keep_indices).item() + int(2*(batch_size - torch.sum(keep_indices).item())/3)
    # to_turn_off = int(1*(batch_size - torch.sum(keep_indices).item())/3)
    # print("Desired drop count: ", to_turn_off)
    # idx = 0
    # # while to_turn_on > 0:
    # #     #print(keep_indices[idx].item())
    # #     if not keep_indices[idx].item():
    # #         #print("Now change it")
    # #         to_turn_on -= 1
    # #         keep_indices[idx] = torch.tensor(True)
    # #     idx += 1
    # psum = torch.cumsum(~keep_indices,0)
    # print(psum)
    # print(psum[psum <= to_turn_off])
    # to_keep_after = torch.sum(psum <= to_turn_off).item()
    # print("To Keep After: ", to_keep_after)
    # keep_indices[to_keep_after:] = torch.Tensor([True]*(batch_size - to_keep_after))
    # print("Final Drop Count:", batch_size - torch.sum(keep_indices).item())
    # print(keep_indices)
    # break
   

print("Total data:", n_val)
print("target=0 data:", n0)
