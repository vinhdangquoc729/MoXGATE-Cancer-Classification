""" 
Experiment script to train and compare 3 models (Gated, Average, Learnable Weights)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import os
from os.path import isfile

# Import 3 phiên bản model khác nhau
from models_1 import multiGCNEncoder as Model_Gated
from models_2 import multiGCNEncoder as Model_Average
from models_3 import multiGCNEncoder as Model_Learnable
from utils import gen_adj_mat_tensor

# --- CẤU HÌNH ---
cuda = True if torch.cuda.is_available() else False
dataType = ['RNA' ,'miRNA','CN','meth'] 
data_fold_train ='./processed_data_common/all/' 
num_class = 32 
epochs = 350
lr_e = 5e-4
dropout = 0.1

# --- HÀM XỬ LÝ DỮ LIỆU ---
def prepare_train_data(data_type, folder = data_fold_train):
    fea_save_file = os.path.join(folder, data_type + '.csv')
    if isfile(fea_save_file):
        df = pd.read_csv(fea_save_file, header=0, index_col=0)
    df = df.T 
    return torch.FloatTensor(df.values.astype(float))

def gen_trte_adj_mat(data_tr, num_class=6):
    adj_metric = "cosine" 
    return gen_adj_mat_tensor(data_tr, num_class, adj_metric)

# --- LOAD DỮ LIỆU CHUNG ---
print("Đang tải dữ liệu...")
omic_dic = {} 
graph_dic = {} 
for dtype in dataType:
    omic_dic[dtype] = prepare_train_data(dtype, data_fold_train)
    graph_dic[dtype] = gen_trte_adj_mat(omic_dic[dtype], num_class)

fea_protein_file = data_fold_train + 'protein.csv'
df_protein = pd.read_csv(fea_protein_file, header=0, index_col=0) 
if df_protein.shape[0] < df_protein.shape[1]:
    df_protein = df_protein.T
protein = torch.FloatTensor(df_protein.values.astype(float))

# Chia tập Train/Val (Cố định seed để so sánh công bằng)
torch.manual_seed(42)
num_samples = omic_dic['RNA'].shape[0]
indices = torch.randperm(num_samples)
split = int(0.8 * num_samples)
train_indices = indices[:split]
val_indices = indices[split:]

if cuda:
    train_indices, val_indices = train_indices.cuda(), val_indices.cuda()
    protein = protein.cuda()
    for dtype in dataType:
        omic_dic[dtype] = omic_dic[dtype].cuda()
        graph_dic[dtype] = graph_dic[dtype].cuda()

# --- CHUẨN BỊ THỬ NGHIỆM ---
dim_list = [omic_dic[x].shape[1] for x in dataType] 
dim_hid_list = [800, 200, 800, 800] 
dim_final = protein.shape[1]

experiments = [
    {"name": "Gated Fusion (Model 1)", "class": Model_Gated, "color": "blue"},
    {"name": "Simple Average (Model 2)", "class": Model_Average, "color": "green"},
    {"name": "Learnable Weights (Model 3)", "class": Model_Learnable, "color": "red"}
]

history_val_loss = {}
best_results = {}

# --- VÒNG LẶP THỬ NGHIỆM ---
for exp in experiments:
    print(f"\n>>> Đang luyện mô hình: {exp['name']}")
    model = exp['class'](dim_list, dim_hid_list, dim_final, dropout, num_samples)
    if cuda: model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_e)
    criterion = torch.nn.MSELoss()
    
    val_losses = []
    best_val = float('inf')
    
    for i in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        z = model(omic_dic['RNA'], omic_dic['meth'], omic_dic['CN'], omic_dic['miRNA'], 
                  graph_dic['RNA'], graph_dic['meth'], graph_dic['CN'], graph_dic['miRNA'])
        loss_train = torch.sqrt(criterion(z[train_indices], protein[train_indices]))
        loss_train.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            z_val = model(omic_dic['RNA'], omic_dic['meth'], omic_dic['CN'], omic_dic['miRNA'], 
                          graph_dic['RNA'], graph_dic['meth'], graph_dic['CN'], graph_dic['miRNA'])
            loss_val = torch.sqrt(criterion(z_val[val_indices], protein[val_indices])).item()
            val_losses.append(loss_val)
            if loss_val < best_val:
                best_val = loss_val
        
        if (i + 1) % 50 == 0:
            print(f"Epoch {i+1:03d} | Train RMSE: {loss_train.item():.6f} | Val RMSE: {loss_val:.6f}")
            
    history_val_loss[exp['name']] = val_losses
    best_results[exp['name']] = best_val

# --- HIỂN THỊ KẾT QUẢ & VẼ BIỂU ĐỒ ---
print("\n" + "="*50)
print("TỔNG KẾT LOSS TỐT NHẤT (BEST VAL RMSE)")
print("="*50)
for name, score in best_results.items():
    print(f"- {name:25}: {score:.6f}")
print("="*50)

plt.figure(figsize=(12, 7))
for exp in experiments:
    plt.plot(history_val_loss[exp['name']], label=exp['name'], color=exp['color'], linewidth=1.5)

plt.xlabel('Epoch')
plt.ylabel('Validation RMSE Loss')
plt.title('Comparison of Fusion Strategies on Validation Set')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Lưu ảnh
if not os.path.exists('./loss_png/'):
    os.makedirs('./loss_png/')
time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
save_path = f'./loss_png/experiment_comparison_{time_str}.png'
plt.savefig(save_path)
plt.show()

print(f"\nĐã lưu biểu đồ so sánh tại: {save_path}")