import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil
import math

def make_fold_data(train_fold, train_y_fold, valid_fold, valid_y_fold, fold):
    
    BATCH_SIZE = 128
    valid_batch = 1024

    print(f'Fold {fold}')

    train_tmp = train_fold.copy()
    train_y_tmp = train_y_fold.copy()
    
    del train_tmp[fold]
    del train_y_tmp[fold]

    valid_tmp = valid_fold[fold]
    valid_y_tmp = valid_y_fold[fold]
    print(f'Train fold data: {len(train_tmp)}')

    train_tmps = torch.cat(train_tmp, dim=0)
    train_y_tmps = torch.cat(train_y_tmp, dim=0)

    train_data = TensorDataset(train_tmps, train_y_tmps)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_data = TensorDataset(valid_tmp, valid_y_tmp)
    valid_loader = DataLoader(valid_data, batch_size=valid_batch, shuffle=True)

    
    print(f'Number of train fold in Cross validation:  {len(train_loaders)}')
    print(f'Number of valid fold in Cross validation:  {len(valid_loaders)}')
    
    return train_loader, valid_loader
            

def make_fold_idx(start_train, start_valid, end_train, end_valid, train_data, valid_data):
    train_tmp = np.where( (train_data[:, 0] >= start_train) & (train_data[:, 0] < end_train) )
    valid_tmp = np.where( (valid_data[:, 0] >= start_valid) & (valid_data[:, 0] < end_valid) )
    
    return train_tmp[0], valid_tmp[0]

def data_unified(data, split):
    train_data = pd.read_csv(f'../data_csv/{data}_train_s{split}.csv').iloc[:, 1:].to_numpy()
    valid_data = pd.read_csv(f'../data_csv/{data}_valid_s{split}.csv').iloc[:, 1:].to_numpy()
    
    print('-------------------------------------------------------------------')
    print(f'All training data: {train_data.shape}')
    print(f'All validation data: {valid_data.shape}')
    
    # train data
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1:]

    # valid data
    X_valid = valid_data[:, :-1]
    y_valid = valid_data[:, -1:]
    
    return X_train, y_train, X_valid, y_valid


def main_func(fold_num, X_train, y_train, X_valid, y_valid):
    
    query_idx = np.unique(X_train[:, 0])
    query_num = len(query_idx)
    print(f'train query 개수: {query_num}')
    
    query_idx_valid = np.unique(X_valid[:, 0])
    valid_min = int(np.min(query_idx_valid))
    print(f'valid query 개수: {len(query_idx_valid)}')

    train_each_fold = query_num // fold_num
    train_fold_idx = [train_each_fold]

    valid_each_fold = len(query_idx_valid) // fold_num
    valid_fold_idx = [(valid_each_fold + valid_min)]
    
    for idx in range(2, fold_num):
        train_before = train_fold_idx[len(train_fold_idx) -1 ]
        train_fold_idx.append(train_before + train_each_fold)
        
        valid_before = valid_fold_idx[len(valid_fold_idx) -1 ]
        valid_fold_idx.append(valid_before + valid_each_fold)

    train_fold_idx.append(query_num)
    valid_fold_idx.append(int(np.max(query_idx_valid)+1))
    
    print(f'train fold num: {fold_num} \nfold_query_idx: {train_fold_idx}')
    print(f'valid fold num: {len(query_idx_valid)} \nfold_query_idx: {valid_fold_idx}')
    
    
    train_fold = []
    train_y_fold = []
    valid_fold = []
    valid_y_fold = []

    idx_train, idx_valid = make_fold_idx(0, np.min(query_idx_valid), train_fold_idx[0], valid_fold_idx[0], X_train, X_valid)
    
    tmps = X_train[idx_train]
    train_fold.append(tmps[:, :])
    train_y_fold.append(y_train[idx_train])
    valid_fold.append(X_valid[idx_valid])
    valid_y_fold.append(y_valid[idx_valid])

    cnt1 = train_fold[0].shape[0]
    cnt2 = valid_fold[0].shape[0]
    for idx in range(fold_num-1):
        idx_train, idx_valid = make_fold_idx(train_fold_idx[idx], valid_fold_idx[idx], train_fold_idx[idx+1], valid_fold_idx[idx+1], X_train, X_valid)
        tmps = X_train[idx_train]
        train_fold.append(tmps[:, :])                                                                                                 
        train_y_fold.append(y_train[idx_train])
        valid_fold.append(X_valid[idx_valid])
        valid_y_fold.append(y_valid[idx_valid])
        
        cnt1 += len(idx_train)
        cnt2 += len(idx_valid)

    print(f'Fold 전체 데이터 개수 확인: train data - {cnt1}, valid data - {cnt2}')
    print(f'Real 전체 데이터 개수 확인: train data - {X_train.shape[0]}, valid data - {X_valid.shape[0]}')
    print('-------------------------------------------------------------------')
    
    return train_fold, train_y_fold, valid_fold, valid_y_fold