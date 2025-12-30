import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import random
import os
import B_dataset_loader
import B_models

# --- CẤU HÌNH ---
# Tập trung vào BRCA hoặc giữ nguyên danh sách cũ
CANCER_LIST = ['BRCA', 'BLCA', 'KIRC', 'LUAD', 'SKCM', 'STAD'] 
# THÊM 'pam50' vào danh sách TASKS để huấn luyện phân loại 5 nhóm phân tử của BRCA
TASKS = ['pam50', 'stage', 'n_stage', 'm_stage', 'histological_type']
BATCH_SIZE = 32
EPOCHS = 250             
LEARNING_RATE = 1e-4
PATIENCE = 15            
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42  

# --- HÀM KHÓA NGẪU NHIÊN (FIX RANDOM SEED) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"   [INFO] Random Seed set to: {seed}")

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs_dict, labels in dataloader:
        inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs_dict)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs_dict, labels in dataloader:
            inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
            labels = labels.to(DEVICE)
            
            outputs = model(inputs_dict)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, acc, f1, precision, recall

def run_experiment(use_protein):
    set_seed(SEED)
    DATA_ROOT = 'processed_data_common_no_protein'
    mode_name = "WITH PROTEIN" if use_protein else "NO PROTEIN (4-Omics)"
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {mode_name}")
    print(f"{'='*60}")

    for cancer in CANCER_LIST:
        print(f"\n>>> PROCESSING CANCER: {cancer}")
        
        for task in TASKS:
            # 1. Load Data
            result = B_dataset_loader.load_and_process_data(cancer, task_type=task, use_protein=use_protein,data_root= DATA_ROOT)
            if result is None:
                continue
                
            train_ds, val_ds, input_dims, num_classes = result
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=np.random.seed(SEED))
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            # 2. Khởi tạo Model (Thông số 32 heads như bài báo cho Cross-Attention)
            print(f"   Task: {task} | Classes: {num_classes} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
            model = B_models.MoXGATE(
                input_dims, 
                num_classes=num_classes,
                hidden_dim=256,
                num_heads=32,       
                dropout=0.3         
            ).to(DEVICE)
                        
            criterion = B_models.FocalLoss() 
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
            
            # 3. Training Loop with Early Stopping
            best_val_loss = float('inf')
            best_metrics = {'acc': 0, 'f1': 0, 'prec': 0, 'loss': 0, 'epoch': 0, 'rec': 0}
            early_stop_counter = 0
            
            for epoch in range(EPOCHS):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, val_f1, val_prec, val_recall = evaluate(model, val_loader, criterion)
                
                if val_acc > best_metrics['acc']:
                    best_metrics = {
                        'acc': val_acc, 
                        'f1': val_f1, 
                        'prec': val_prec, 
                        'loss': val_loss,
                        'rec': val_recall,
                        'epoch': epoch + 1
                    }
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0 
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= PATIENCE:
                    print(f"      [STOP] Early stopping tại epoch {epoch+1}.")
                    break
            
            # 4. In kết quả cuối cùng
            print(f"   [RESULT] {cancer} - {task} ({mode_name}):")
            print(f"     -> Best Epoch: {best_metrics['epoch']}")
            print(f"     -> Accuracy  : {best_metrics['acc']:.4f}")
            print(f"     -> F1-Score  : {best_metrics['f1']:.4f}")
            print(f"     -> Precision : {best_metrics['prec']:.4f}")
            print(f"     -> Recall    : {best_metrics['rec']:.4f}")
            print(f"     -> Val Loss  : {best_metrics['loss']:.4f}")

if __name__ == "__main__":
    run_experiment(use_protein=False)
    run_experiment(use_protein=True)