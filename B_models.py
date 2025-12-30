import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    """Mã hóa đặc trưng riêng cho từng loại dữ liệu (RNA, Meth...)"""
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        x = self.linear(x) # (Batch, Hidden)
        x = x.unsqueeze(1) # (Batch, 1, Hidden) - Tạo sequence giả
        
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out)) # Residual + Norm
        return x

class MoXGATE(nn.Module):
    def __init__(self, input_dims_dict, num_classes, hidden_dim=256, num_heads=8, dropout=0.2):
        super().__init__()
        self.modalities = list(input_dims_dict.keys())
        self.encoders = nn.ModuleDict()
        
        # Tạo encoder riêng cho từng loại
        for mod, dim in input_dims_dict.items():
            self.encoders[mod] = ModalityEncoder(dim, hidden_dim, num_heads=8, dropout=dropout)
            
        # Trọng số học được (Learnable Weights) để trộn các modalities
        self.modality_weights = nn.Parameter(torch.ones(len(self.modalities)))
        
        # Cross-Attention Block (Trộn thông tin)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, inputs_dict):
        encoded_list = []
        
        # 1. Encode từng nguồn
        for mod in self.modalities:
            x = inputs_dict[mod]
            encoded_list.append(self.encoders[mod](x)) # Mỗi cái shape (Batch, 1, Hidden)
            
        # 2. Modality Weighted Fusion
        # Ép trọng số về tổng = 1 (Softmax)
        weights = F.softmax(self.modality_weights, dim=0)
        
        # Ghép lại thành chuỗi (Batch, Num_Modalities, Hidden)
        concat_features = torch.cat(encoded_list, dim=1)
        
        # Cross-Attention: Để các modality "nhìn" thấy nhau
        attn_out, _ = self.cross_attn(concat_features, concat_features, concat_features)
        
        # Tính tổng có trọng số (Weighted Sum)
        weighted_out = torch.zeros_like(attn_out[:, 0, :])
        for i in range(len(self.modalities)):
            weighted_out += weights[i] * attn_out[:, i, :]
            
        # 3. Phân loại
        logits = self.classifier(weighted_out)
        return logits

# Hàm Focal Loss (Xử lý mất cân bằng dữ liệu)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)