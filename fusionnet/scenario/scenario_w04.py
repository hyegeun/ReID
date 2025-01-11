import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.backends.cudnn as cudnn
import data_edit
from copy import deepcopy

import os
import shutil
import math
import time
import argparse
import copy
import logging

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


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


def train(train_loader, valid_loader, base_path, input_size, hidden_size1):
    model = MLP(input_size, hidden_size1)

    # Define the loss function and optimizer
    criterion_bce = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Define variables to keep track of the best performing model
    best_accuracy = 0.0
    best_loss = math.inf
    best_model_state_dict = None

    # Training epoch
    num_epochs = 50

    # model Training
    train_loss = []
    valid_loss = []
    valid_accuracy = [] 

    start = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss_bce = criterion_bce(outputs, batch_y)
            total_loss += loss_bce.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss_bce.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)

        # After each epoch, calculate the validation accuracy to determine the best model
        with torch.no_grad():
            model.eval()
            total_loss_val = 0.0
            total_correct = 0
            total_samples = 0

            for batch_X_val, batch_y_val in valid_loader:
                outputs_val = model(batch_X_val)
                loss_bce_val = criterion_bce(outputs_val, batch_y_val)
                total_loss_val += loss_bce_val.item()

                predicted_classes_val = (outputs_val >= 0.5).float()
                total_correct += (predicted_classes_val == batch_y_val).sum().item()
                total_samples += batch_y_val.size(0)

            # print(total_correct, total_samples)
            avg_loss_val = total_loss_val / len(valid_loader)
            valid_loss.append(avg_loss_val)

            # Calculate the validation accuracy for this epoch
            accuracy = total_correct / total_samples
            valid_accuracy.append(accuracy) 

            # Save the model from the last epoch
            torch.save(model.state_dict(), f'{base_path}/last.pth')

            # print(model.fc1.weight)
            # Save the best performing model state dictionary
            # if accuracy >= best_accuracy or avg_loss_val <= best_loss:
            if accuracy > best_accuracy:
                print(epoch+1)
            # if avg_loss_val <= best_loss:
                best_accuracy = accuracy
                best_loss = avg_loss_val
                best_model_state_dict = deepcopy(model.state_dict())
                print(model.fc1.weight)
                # print(model.fc1.bias)


        if (epoch + 1) % 1 == 0:
            msg = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f} Valid Loss: {avg_loss_val:.4f}, Valid Accuracy: {accuracy:.4f}'
            logging.info(msg)
            print(msg)

    end = time.time()
    msg = f'total training time: {end-start:.5f} sec'
    logging.info(msg)
    print(msg)

    # Save the best performing model after all epochs are completed
    torch.save(best_model_state_dict, f'{base_path}/best.pth')
    msg = f'Best model saved at {base_path}/best.pth'
    logging.info(msg)
    print(msg)

    logging.shutdown()
    
    
def data_processing(method, fold_num, train_fold, train_y_fold, valid_fold, valid_y_fold):
    
    # histogram file
    hist_csv = pd.read_csv(method)
    histogram = hist_csv.iloc[:, 2:].to_numpy()
    bin_num = histogram.shape[1]
    
    same_cam = np.full((1, bin_num), 1/bin_num, dtype=np.float32)
    
    histogram = np.vstack((histogram, same_cam))
    
    for fold in range(fold_num):
        msg = f'Fold {fold} data + histogram'
        logging.info(msg)
        # print(msg)
        
        # 해당 fold에 해당하는 데이터들 가져오기 
        X_train = train_fold[fold]
        y_train = train_y_fold[fold]
        X_valid = valid_fold[fold]
        y_valid = valid_y_fold[fold]
        
        # train data에 histogram 붙이는 과정 
        bin_idx = np.array(X_train[:, 8], dtype=np.int32).squeeze() # Data가 가진 bin index 
        row_idx = np.array(X_train[:, -1:], dtype=np.int32).squeeze()   # Data의 camera pair index 
        all_hist = histogram[row_idx] #각 데이터에 대한 histogram 

        ext_hist = np.zeros((all_hist.shape[0], window_size*2+1)) # window size에 따라 달라지는 attention histogram?

        for row in range(all_hist.shape[0]):
            left_bin = bin_idx[row]-window_size
            right_bin = bin_idx[row]+window_size

            if left_bin >= 0 and right_bin < bin_num:
                ext_hist[row] = all_hist[row, left_bin : right_bin+1]

            elif left_bin < 0:
                extra = 0 - left_bin
                ext_hist[row][extra:] = all_hist[row, 0 : right_bin+1]

            elif left_bin < bin_num and right_bin > bin_num:
                extra = bin_num - left_bin
                ext_hist[row][0 : extra] = all_hist[row, left_bin : bin_num]

        ext_hist = torch.tensor(ext_hist, dtype=torch.float32)
        ext_hist = ext_hist.squeeze()

        data_x = torch.tensor(X_train[:, 7:-2], dtype=torch.float32)

        msg = f'Initial Train data : {data_x.shape}'
        logging.info(msg)
        print(msg)
        
        msg = f'histogram dimension: {ext_hist.shape}'
        logging.info(msg)
        print(msg)
        
        X_trains = torch.cat([data_x, ext_hist], dim=1)
        
        msg = f'(except query idx at index 0) Train data shape: {X_trains.shape}'
        logging.info(msg)
        print(msg)

        # valid data에 histogram 붙이는 과정 
        bin_idx = np.array(X_valid[:, 8], dtype=np.int32).squeeze() # Data가 가진 bin index 
        row_idx = np.array(X_valid[:, -1:], dtype=np.int32).squeeze()
        all_hist = histogram[row_idx] #각 데이터에 대한 histogram 

        ext_hist = np.zeros((all_hist.shape[0], window_size*2+1)) # window size에 따라 달라지는 attention histogram?

        for row in range(all_hist.shape[0]):
            left_bin = bin_idx[row]-window_size
            right_bin = bin_idx[row]+window_size

            if left_bin >= 0 and right_bin < bin_num:
                ext_hist[row] = all_hist[row, left_bin : right_bin+1]

            elif left_bin < 0:
                extra = 0 - left_bin
                ext_hist[row][extra:] = all_hist[row, 0 : right_bin+1]

            elif left_bin < bin_num and right_bin > bin_num:
                extra = bin_num - left_bin
                ext_hist[row][0 : extra] = all_hist[row, left_bin : bin_num]

        ext_hist = torch.tensor(ext_hist, dtype=torch.float32)
        ext_hist = ext_hist.squeeze()

        data_x = torch.tensor(X_valid[:, 7:-2], dtype=torch.float32)
        X_valids = torch.cat([data_x, ext_hist], dim=1)

        print(f'Initial Valid data : {data_x.shape}')
        print(f'histogram dimension: {ext_hist.shape}')
        print(f'(except query idx at index 0) Valid data shape: {X_valids.shape}')
        print('-------------------------------------------------------------------')
        
        train_fold[fold] = X_trains
        train_y_fold[fold] = torch.tensor(y_train, dtype=torch.float32)
        valid_fold[fold] = X_valids
        valid_y_fold[fold] = torch.tensor(y_valid, dtype=torch.float32)
        
    train_x = torch.tensor(np.vstack(train_fold))
    train_y = torch.tensor(np.vstack(train_y_fold))
    valid_x = torch.tensor(np.vstack(valid_fold))
    valid_y = torch.tensor(np.vstack(valid_y_fold))
    
    return train_x, train_y, valid_x, valid_y
        
        
print ("PyTorch version:[%s]."%(torch.__version__)) # 토치 버전 확인
# device에 일반 GPU or M1 GPU or CPU를 할당해주는 코드
if torch.cuda.is_available() : # 일반 GPU 사용시
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
elif torch.backends.mps.is_available(): # 맥 M1 GPU 사용시
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print ("device:[%s]."%(device))

#인자값 
parser = argparse.ArgumentParser()  
parser.add_argument('--method', default ='../parzen_window_csv/df_parzen_window_5.csv')
parser.add_argument('--base_path', default ='')
parser.add_argument('--fold_num', type=int, default = 5)
parser.add_argument('--data', default ='')
parser.add_argument('--seed', type=int, default = 20230807)
parser.add_argument('--split', type=int, default = 1)
opt = parser.parse_args()

# 파이썬 랜덤시드 고정
seed = opt.seed
deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# model Training path
if os.path.isdir(opt.base_path):
    shutil.rmtree(opt.base_path)
    os.makedirs(opt.base_path)
else:
    os.makedirs(opt.base_path)

# Log file 생성 
log_file = os.path.join(opt.base_path, "log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


with ClearCache():
    window_size = 4 # window size 절반 
    input_size = 1 + window_size*2 + 1 # 10 
    hidden_size1 = int(input_size * 2/ 3) + 1
    BATCH_SIZE = 64
    valid_batch = 4096
    
    X_train, y_train, X_valid, y_valid = data_edit.data_unified(opt.data, opt.split) # 모든 데이터를 train / valid로 이쁘게 만들어놓고 
    
    # 폴드 수에 따라 데이터를 폴드로 나눠놓기 (프로세스가 죽어서 나눠서 진행) 
    train_fold, train_y_fold, valid_fold, valid_y_fold = data_edit.main_func(opt.fold_num, X_train, y_train, X_valid, y_valid)
    
    # 폴드로 나눠서 데이터를 처리하고 다시 합침 
    train_x, train_y, valid_x, valid_y = data_processing(opt.method, opt.fold_num, train_fold, train_y_fold, valid_fold, valid_y_fold)

    train_data = TensorDataset(train_x, train_y)
    valid_data = TensorDataset(valid_x, valid_y)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=valid_batch, shuffle=True)

    msg = '-------------------------------------------------------------------'
    logging.info(msg)
    print(msg)

    train(train_loader, valid_loader, opt.base_path, input_size, hidden_size1)
