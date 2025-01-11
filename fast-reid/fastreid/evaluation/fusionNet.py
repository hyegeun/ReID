import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.backends.cudnn as cudnn
import math

import copy
import logging


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    
def processing(dist_mat, hist_idx_list, row_idx_list, parzen, window_size):
    topology_csv = pd.read_csv(parzen)
    topology = topology_csv.iloc[:, 2:].to_numpy()
    bin_num = topology.shape[1]
    
    same_cam = np.full((1, bin_num), 1/bin_num, dtype=np.float32) # cam이 같은 경우 
    topology = np.vstack((topology, same_cam)) # 기존에 cam이 같은 경우를 합쳐줌 
    
    data_num = dist_mat.shape[0] # number of data
    
    r_iter = 10000
    r_size = data_num // r_iter
    
    topology_list = [] # topology list
    
    for r_idx in range(0, r_iter+1):
        start = r_size * r_idx
        
        if r_idx == r_iter:
            end = data_num
        else:
            end = r_size * (r_idx +1)
        
        r_hist_idx = hist_idx_list[start:end] # 해당 data들에 대한 bin index 
        r_row_idx = row_idx_list[start:end]   # 해당 data들에 대한 camera pair index 
        r_topology = topology[r_row_idx]      # camera topology에서 data들에 해당하는 index의 topology들을 가져옴 
        topology_ext = np.zeros((r_topology.shape[0], window_size*2+1))
#         topology_ext = np.full((r_topology.shape[0], window_size*2+1), 1/bin_num)
        
        if window_size > 0:
            for row in range(r_topology.shape[0]):
                left_bin = r_hist_idx[row]-window_size
                right_bin = r_hist_idx[row]+window_size

                if left_bin >= 0 and right_bin < bin_num:
                    topology_ext[row] = r_topology[row, left_bin : right_bin+1]

                elif left_bin < 0:
                    extra = 0 - left_bin
                    topology_ext[row][extra:] = r_topology[row, 0 : right_bin+1]

                elif left_bin < bin_num and right_bin > bin_num:
                    extra = bin_num - left_bin
                    topology_ext[row][0 : extra] = r_topology[row, left_bin : bin_num]
        
        else:
            for row in range(r_topology.shape[0]):
                bins = r_hist_idx[row]

                if bins < bin_num:
                    topology_ext[row] = bins
                else:
                    topology_ext[row] = 0
        
        topology_list.append(topology_ext)
        
        if r_idx % 1000 == 0:
            print(f'{r_idx}/{r_iter} processing.')
        
    all_topology = np.vstack(topology_list)
    print(f'dist_mat shape = {dist_mat.shape}')
    print(f'all_topology shape = {all_topology.shape}')
    topology_data = np.hstack([dist_mat, all_topology])
    print(f'topology data shape = {topology_data.shape}')
    
    fusion_data = torch.tensor(topology_data, dtype=torch.float32)
    
    return fusion_data
        
        
def making(query_camids, query_frames, gallery_camids, gallery_frames, max_cam):
    
    q_num = len(query_camids)
    g_num =len(gallery_camids)
    
    # 계산 편의를 위해 카메라 번호를 다시 0 ~ max_cam-1에서 1 ~ max_cam으로 변환 
    query_camids = query_camids + 1
    gallery_camids = gallery_camids + 1
    
    hist_idx_list = []
    row_idx_list = []
    
    for q_idx in range(q_num):
        q_camid = query_camids[q_idx]
        q_frame = query_frames[q_idx]
        
        for g_idx in range(g_num):
            g_camid = gallery_camids[g_idx]
            g_frame = gallery_frames[g_idx]
            
            if q_frame < g_frame:
                    cam1 = q_camid
                    cam2 = g_camid
            else:
                cam2 = q_camid
                cam1 = g_camid

            if cam1 < cam2:
                row_idx = (cam1-1) * max_cam + cam2 - cam1 - 1

            elif cam1 > cam2:
                row_idx = (cam1-1) * max_cam + cam2 - cam1

            elif cam1 == cam2:
                row_idx = max_cam**2 - max_cam

            delta_frame = abs(q_frame - g_frame)
            hist_idx = delta_frame // 100
            
            hist_idx_list.append(hist_idx)
            row_idx_list.append(row_idx)
            
    print('hist idx and row idx calculated.')
    
    return hist_idx_list, row_idx_list
                

def main(dist, query_camids, query_frames, gallery_camids, gallery_frames, fusion_model_path, parzen):
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    data_name = fusion_model_path.split('/')[-4]
    print(f'Dataset = {data_name}')

    if data_name == 'veri':
        max_cam = 20 
    else:
        max_cam = 6
    print(f'max_cam: {max_cam}')
    
    print(f'fusion_model = {fusion_model_path}')
    print(f'Parzen path = {parzen}')
    
    windows = fusion_model_path.split('/')[-3]
    windows = windows.split('_')[1]
    window = int(windows[1:])
    print(f'Window size = {window}')
    input_size = 1 + window*2 + 1
    hidden_size1 = int(input_size * 2 / 3) + 1

    query_num, gallery_num = dist.shape
    batch_size = gallery_num * 100
    
    dist_mat = np.reshape(dist, (-1, 1))
    cos_prob = (2 - dist_mat) / 2
    
    hist_idx_list, row_idx_list = making(query_camids, query_frames, gallery_camids, gallery_frames, max_cam)
    fusion_data = processing(cos_prob, hist_idx_list, row_idx_list, parzen, window)
    test_loader = DataLoader(fusion_data, batch_size = batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    fusion_model = MLP(input_size, hidden_size1).to(device)
    fusion_model.load_state_dict(torch.load(fusion_model_path))
    fusion_model.eval()
    
    fusion_prob_list = []

    print('fusion model eval start...')
    d_num = query_num * gallery_num
    for i_idx, data in enumerate(test_loader):
        try:
            output = fusion_model(data.to(device)).cpu()
            output_np = output.detach().numpy()
            output_np.reshape(-1, 1)
            fusion_prob_list.append(output_np)        

            if i_idx % int(d_num/batch_size * 0.1) == 0:
                print(f'{i_idx}/{d_num/batch_size} fusion model...')
                
        except Exception as e:
            print(i_idx, e)
            print(output_np.shape)
            print(type(output_np))
        
    fusion_prob = np.vstack(fusion_prob_list)
    print(f'fusion_prob shape = {fusion_prob.shape}')
    fusion_prob_reshape = np.reshape(fusion_prob, (query_num, gallery_num))
    print(f'fusion_prob_reshape shape = {fusion_prob_reshape.shape}')
    
    fusion_dist = 2 - (fusion_prob_reshape*2)
    
    return fusion_dist