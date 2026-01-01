import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

# --- GCN Layer (Giữ nguyên) ---
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
          
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight) 
        output = torch.sparse.mm(adj, support) 
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# --- GraphGCN Model (Giữ nguyên) ---
class GraphGCN(nn.Module):
    def __init__(self, nfeat, nhid, dim_final, dropout, npatient):
        super(GraphGCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(nfeat, nhid)        
        self.gc2 = GraphConvolution(nhid , dim_final)   

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z

class GatedFusion(nn.Module):
    """
    Cơ chế Gated Fusion: Học trọng số động cho từng modality.
    z = alpha_1 * h_1 + alpha_2 * h_2 + ...
    Trong đó alpha được học từ chính h bằng mạng Neural.
    """
    def __init__(self, input_dim, num_modalities=4):
        super(GatedFusion, self).__init__()
        # Mạng học trọng số Attention (Gate)
        # Input: input_dim (256), Output: 1 (score)
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1) 
            ) for _ in range(num_modalities)
        ])
        
    def forward(self, x_list):
        # x_list: List gồm 4 tensor [x_rna, x_meth, x_cn, x_mic]
        # Mỗi tensor shape: (Batch, Dim)
        
        # 1. Tính điểm quan trọng (Attention Score) cho từng nguồn
        scores = []
        for i, x in enumerate(x_list):
            s = self.gate_nets[i](x) # (Batch, 1)
            scores.append(s)
            
        # 2. Chuyển điểm số thành trọng số (Softmax) để tổng = 1
        scores = torch.cat(scores, dim=1) # (Batch, 4)
        weights = F.softmax(scores, dim=1) # (Batch, 4)
        
        # 3. Tính tổng có trọng số (Weighted Sum)
        # weights[:, 0] là trọng số của RNA cho từng bệnh nhân
        fused = torch.zeros_like(x_list[0])
        for i, x in enumerate(x_list):
            # Mở rộng trọng số để nhân: (Batch, 1) * (Batch, Dim)
            w = weights[:, i].unsqueeze(1) 
            fused += w * x
            
        return fused

class multiGCNEncoder(nn.Module):
    def __init__(self, dim_list, dim_hid_list, dim_final, dropout, npatient):
        super(multiGCNEncoder, self).__init__()
        self.dropout = dropout
        self.hidden_dim = 256 # Kích thước chung sau khi qua GCN
        
        # 1. Feature Extractors (GCNs)
        self.RNAseq = GraphGCN(dim_list[0], dim_hid_list[0], self.hidden_dim, dropout, npatient)
        self.miRNA = GraphGCN(dim_list[1], dim_hid_list[1], self.hidden_dim, dropout, npatient)
        self.CN = GraphGCN(dim_list[2], dim_hid_list[2], self.hidden_dim, dropout, npatient)
        self.meth = GraphGCN(dim_list[3], dim_hid_list[3], self.hidden_dim, dropout, npatient)

        # 2. Gated Fusion Mechanism (Thay cho cộng/nối)
        # self.fusion = GatedFusion(input_dim=self.hidden_dim, num_modalities=4)
        
        # 3. Final Prediction Head
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, dim_final) # Output ra 455 Protein
        )

    def forward(self, RNAseq, dnam, cn, mic, adj_1, adj_2, adj_3, adj_4):
        # Bước 1: Trích xuất đặc trưng
        x_r = self.RNAseq(RNAseq, adj_1)
        x_d = self.meth(dnam, adj_2)
        x_c = self.CN(cn, adj_3)
        x_mic = self.miRNA(mic, adj_4)
        
        # Bước 2: Hợp nhất
        z_fused = (x_r + x_d + x_c + x_mic) / 4.0
        # Bước 3: Dự đoán cuối cùng
        z = self.predictor(z_fused)
        
        return z