'''
Import the trained translation model and apply it to 3085 omics samples to generate corresponding proteomics.
'''

import numpy as np
import pandas as pd
import torch
from models import multiGCNEncoder
from utils import  gen_adj_mat_tensor
from os.path import isfile

cuda = True if torch.cuda.is_available() else False
dataType = ['RNA' ,'miRNA','CN','meth'] 
data_fold ='./processed_data_common/all/'      #samples = 3851


def prepare_train_data(data_type): 
    file_input = data_fold
    fea_save_file = file_input + data_type + '.csv'
    if isfile(fea_save_file):
        df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)   
    
    df = df.T # Transpose: Hàng là bệnh nhân, Cột là features
    
    # Lấy index (tên bệnh nhân) ra ngoài
    patient_names = df.index 
    
    tensor = torch.FloatTensor(df.values.astype(float))
    return tensor, patient_names


def gen_trte_adj_mat(data_tr,num_class=6):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_train_list.append(gen_adj_mat_tensor(data_tr, num_class, adj_metric))

    return adj_train_list 


omic_dic ={} 
graph_dic ={} 
num_class = 32 
saved_patient_names = None # Biến để lưu tên bệnh nhân

############## load data/graph/model #####################
print("\n############## load model ...##############\n")
for type in dataType:
    # Nhận thêm tên bệnh nhân từ hàm đã sửa
    tensor_data, patients = prepare_train_data(type)
    omic_dic[type] = tensor_data
    
    # Lưu lại tên bệnh nhân từ loại dữ liệu đầu tiên (ví dụ RNA)
    if saved_patient_names is None:
        saved_patient_names = patients
        
    graph_dic[type] = gen_trte_adj_mat(omic_dic[type], num_class)

dim_list = [omic_dic[x].shape[1] for x in dataType]
dim_hid_list = [800,200,800,800]
dim_final = 455
dropout = 0.1
npatient = omic_dic['RNA'].shape[0]
model = multiGCNEncoder(dim_list, dim_hid_list,dim_final, dropout,npatient)
model.load_state_dict(torch.load('model.pth'))

if cuda:
    model.cuda()
for type in dataType:
    if cuda:
        omic_dic[type]=omic_dic[type].cuda()
        graph_dic[type]=graph_dic[type][0].cuda()
    else:
        graph_dic[type]=graph_dic[type][0]

############## generate protein #######################
print("\n############## computing start ...##############\n")

protein = model(RNAseq=omic_dic['RNA'], dnam=omic_dic['meth'], cn=omic_dic['CN'], mic=omic_dic['miRNA'], adj_1=graph_dic['RNA'], adj_2=graph_dic['meth'], adj_3=graph_dic['CN'], adj_4=graph_dic['miRNA'])

print("\n############## computing end ...##############\n")

protein_ref_file = data_fold + 'protein.csv' # Hoặc file nào chứa danh sách protein chuẩn
protein_columns = None
############## save protein data #######################
if isfile(protein_ref_file):
    # Đọc file protein gốc để lấy tên cột (Features)
    # Lưu ý: Nếu file gốc cũng cần Transpose như các file kia thì phải .T
    # Giả sử file gốc: Index=Protein, Cols=Patient -> Đọc xong index là Protein
    temp_df = pd.read_csv(protein_ref_file, sep=',', header=0, index_col=0)
    protein_columns = temp_df.index # Lấy tên Protein
else:
    print("Warning: Không tìm thấy file protein mẫu để lấy tên cột. Sẽ dùng index số.")

# 2. Tạo DataFrame kết quả với đầy đủ tên
result_df = pd.DataFrame(
    protein.detach().cpu().numpy(), 
    index=saved_patient_names,  # Gán tên bệnh nhân vào hàng
    columns=protein_columns     # Gán tên protein vào cột (nếu có)
)

# 3. Nếu file gốc bị ngược (Protein là hàng), cần Transpose lại cho đúng chuẩn output
# Kết quả mong muốn: Hàng = Protein, Cột = Bệnh nhân (để giống định dạng file .fea gốc)
result_df = result_df.T 

result_df.to_csv(data_fold + 'protein_generate.fea', sep=',')
print("Saved with metadata!")
# Predicted protein data of #3851 samples
