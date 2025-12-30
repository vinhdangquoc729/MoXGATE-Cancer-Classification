""" 
  Training of the module A to obtain the translation model, and output the model as model.pth
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import multiGCNEncoder
from utils import gen_adj_mat_tensor
from os.path import isfile
import datetime
import os

# --- CẤU HÌNH ---
cuda = True if torch.cuda.is_available() else False
# cuda = False
dataType = ['RNA' ,'miRNA','CN','meth'] 
data_fold_train ='./processed_data_common/all/' 
data_fold_infer = './processed_data_common_no_protein/all/'
# --- HÀM XỬ LÝ DỮ LIỆU ---

def prepare_train_data(data_type, folder = data_fold_train): # Thêm tham số folder
    fea_save_file = os.path.join(folder, data_type + '.csv')
    if isfile(fea_save_file):
        df = pd.read_csv(fea_save_file, header=0, index_col=0)
    df = df.T 
    return torch.FloatTensor(df.values.astype(float))

def gen_trte_adj_mat(data_tr, num_class=6):
    adj_metric = "cosine" 
    adj_train_list = []
    adj_train_list.append(gen_adj_mat_tensor(data_tr, num_class, adj_metric)) 
    return adj_train_list 

# --- LOAD DỮ LIỆU ---
omic_dic = {} 
graph_dic = {} 
num_class = 32 

print("Đang tải dữ liệu...")
for type in dataType:
    omic_dic[type] = prepare_train_data(type, data_fold_train)
    # Tạo đồ thị dựa trên TOÀN BỘ dữ liệu (để giữ cấu trúc liên kết tốt nhất)
    graph_dic[type] = gen_trte_adj_mat(omic_dic[type], num_class)

# Load Protein (Target)
fea_protein_file = data_fold_train + 'protein.csv'
df_protein = pd.read_csv(fea_protein_file, header=0, index_col=0) 
# Xử lý chiều dữ liệu protein cho khớp
if df_protein.shape[0] < df_protein.shape[1]:
    df_protein = df_protein.T
protein = torch.FloatTensor(df_protein.values.astype(float))
protein_names = df_protein.columns

print(f"Dữ liệu đầu vào (RNA): {omic_dic['RNA'].shape}")
print(f"Dữ liệu đích (Protein): {protein.shape}")

# --- CHIA TẬP TRAIN / VAL (80/20) ---
num_samples = omic_dic['RNA'].shape[0]
indices = torch.randperm(num_samples) # Xáo trộn ngẫu nhiên

split = int(0.8 * num_samples) # Điểm cắt 80%
train_indices = indices[:split]
val_indices = indices[split:]

print(f"Số lượng Train: {len(train_indices)} | Số lượng Val: {len(val_indices)}")

# --- KHỞI TẠO MODEL ---
dim_list = [omic_dic[x].shape[1] for x in dataType] 
dim_hid_list = [800,200,800,800] 
dim_final = protein.shape[1] 

dropout = 0.1   
lr_e = 5e-4
npatient = num_samples # Tổng số bệnh nhân

model = multiGCNEncoder(dim_list, dim_hid_list, dim_final, dropout, npatient)
optim = torch.optim.Adam(list(model.parameters()), lr=lr_e) 
criterion = torch.nn.MSELoss()

# --- CHUYỂN SANG GPU (NẾU CÓ) ---
if cuda:
    model.cuda()
    train_indices = train_indices.cuda()
    val_indices = val_indices.cuda()
    protein = protein.cuda()
    for type in dataType:
        omic_dic[type] = omic_dic[type].cuda()
        graph_dic[type] = graph_dic[type][0].cuda()
else:
    for type in dataType:
        graph_dic[type] = graph_dic[type][0]

# --- VÒNG LẶP HUẤN LUYỆN ---
epoch = 350
train_loss_list = []
val_loss_list = []

# torch.autograd.set_detect_anomaly(True) 
print("\n############## Bắt đầu training ...##############\n")

for i in range(epoch):
    # 1. Training Step
    model.train()
    optim.zero_grad()
    
    # Forward pass (Chạy trên toàn bộ graph)
    z = model(
        RNAseq=omic_dic['RNA'], dnam=omic_dic['meth'], cn=omic_dic['CN'], mic=omic_dic['miRNA'], 
        adj_1=graph_dic['RNA'], adj_2=graph_dic['meth'], adj_3=graph_dic['CN'], adj_4=graph_dic['miRNA']
    )
    
    # Tính Loss CHỈ trên tập Train
    loss_train = criterion(z[train_indices], protein[train_indices])
    loss_train = torch.sqrt(loss_train) # RMSE
    
    loss_train.backward()
    optim.step()
    
    train_loss_list.append(loss_train.item())

    # 2. Validation Step (Không tính gradient)
    model.eval()
    with torch.no_grad():
        # Forward pass lại (hoặc dùng lại z nếu bộ nhớ cho phép, nhưng để an toàn gọi lại ở chế độ eval)
        z_val = model(
            RNAseq=omic_dic['RNA'], dnam=omic_dic['meth'], cn=omic_dic['CN'], mic=omic_dic['miRNA'], 
            adj_1=graph_dic['RNA'], adj_2=graph_dic['meth'], adj_3=graph_dic['CN'], adj_4=graph_dic['miRNA']
        )
        # Tính Loss CHỈ trên tập Val
        loss_val = criterion(z_val[val_indices], protein[val_indices])
        loss_val = torch.sqrt(loss_val)
        val_loss_list.append(loss_val.item())

    # In thông tin mỗi 10 epoch
    if (i + 1) % 10 == 0:
        print(f"Epoch {i+1:04d} | Train Loss: {loss_train.item():.6f} | Val Loss: {loss_val.item():.6f}")

print("############## Kết thúc ##############")

# --- LƯU KẾT QUẢ ---
# Lưu file dự đoán (Lấy kết quả dự đoán của toàn bộ dữ liệu)
print(f"\n--- Đang nạp dữ liệu từ {data_fold_infer} để dự đoán ---")
patient_names_infer = pd.read_csv(os.path.join(data_fold_infer, 'RNA.csv'), index_col=0).columns
for type in dataType:
    # Load lại dữ liệu genomics của 3851 bệnh nhân [cite: 96, 182]
    omic_dic[type] = prepare_train_data(type, data_fold_infer) 
    
    # Tạo lại ma trận kề (graph) cho tập bệnh nhân mới [cite: 140, 147]
    graph_dic[type] = gen_trte_adj_mat(omic_dic[type], num_class)

# Cập nhật số lượng bệnh nhân cho mô hình GCN [cite: 140, 160]
model.npatient = omic_dic['RNA'].shape[0] 
if cuda:
    for type in dataType:
        omic_dic[type] = omic_dic[type].cuda()
        graph_dic[type] = graph_dic[type][0].cuda()
else:
    for type in dataType:
        graph_dic[type] = graph_dic[type][0]

# --- BẮT ĐẦU DỰ ĐOÁN TRÊN DỮ LIỆU MỚI ---
FOLDERS = ['all', 'BLCA', 'BRCA', 'KIRC', 'LUAD', 'SKCM', 'STAD', 'UCEC', 'UVM']
DATA_ROOT_INFER = './processed_data_common_no_protein/'

print(f"\n############## Bắt đầu dự đoán cho từng thư mục ...##############")

model.eval()
with torch.no_grad():
    for folder in FOLDERS:
        current_infer_path = os.path.join(DATA_ROOT_INFER, folder)
        if not os.path.exists(current_infer_path):
            print(f"-> Bỏ qua: {folder} (Không tìm thấy thư mục)")
            continue

        print(f"-> Đang xử lý: {folder}...")
        
        # 1. Lấy tên bệnh nhân của bộ dữ liệu hiện tại
        patient_names = pd.read_csv(os.path.join(current_infer_path, 'rna.csv'), index_col=0).columns
        
        # 2. Nạp dữ liệu omics và tạo đồ thị cho folder hiện tại
        current_omic_dic = {}
        current_graph_dic = {}
        for dtype in dataType:
            current_omic_dic[dtype] = prepare_train_data(dtype, current_infer_path)
            # Tạo ma trận kề dựa trên độ tương đồng Cosine [cite: 140, 144]
            current_graph_dic[dtype] = gen_trte_adj_mat(current_omic_dic[dtype], num_class)[0]

        # 3. Cập nhật số lượng bệnh nhân cho mô hình GCN [cite: 160]
        model.npatient = current_omic_dic['RNA'].shape[0]
        
        if cuda:
            for dtype in dataType:
                current_omic_dic[dtype] = current_omic_dic[dtype].cuda()
                current_graph_dic[dtype] = current_graph_dic[dtype].cuda()

        # 4. Dự đoán Protein (Translation Phase) [cite: 78, 164]
        z_out = model(
            RNAseq=current_omic_dic['RNA'], dnam=current_omic_dic['meth'], 
            cn=current_omic_dic['CN'], mic=current_omic_dic['miRNA'], 
            adj_1=current_graph_dic['RNA'], adj_2=current_graph_dic['meth'], 
            adj_3=current_graph_dic['CN'], adj_4=current_graph_dic['miRNA']
        )

        # 5. Lưu file protein_new.fea vào từng thư mục tương ứng
        output_file = os.path.join(current_infer_path, 'protein_new.fea')
        pd.DataFrame(z_out.detach().cpu().numpy(), 
                     index=patient_names, 
                     columns=protein_names).to_csv(output_file, sep='\t')
        print(f"   => Đã lưu: {output_file}")

# --- VẼ BIỂU ĐỒ (2 đường) ---
plt.figure(figsize=(10, 6))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('RMSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

# Tạo thư mục lưu ảnh nếu chưa có
if not os.path.exists('./loss_png/'):
    os.makedirs('./loss_png/')

now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{time_str}.png"
plt.savefig('./loss_png/'+ file_name)
print(f"Đã lưu biểu đồ loss tại ./loss_png/{file_name}")