import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import B_dataset_loader
import B_models

CANCER_LIST = ['BRCA', 'LUAD', 'BLCA', 'KIRC', 'SKCM'] 
TASKS = ['pam50', 'stage', 'n_stage', 'm_stage', 'histological_type']
BATCH_SIZE = 32
EPOCHS = 250             
LEARNING_RATE = 5e-5
PATIENCE = 15            
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
RESULT_DIR = "experimental_results"
os.makedirs(RESULT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"   [INFO] Random Seed set to: {seed}")

def save_split_evidence(train_idx, val_idx, test_idx, cancer, task, mode):
    """Lưu danh sách chỉ số (indices) của các tập để làm minh chứng thực nghiệm"""
    split_data = {
        'split': ['train']*len(train_idx) + ['val']*len(val_idx) + ['test']*len(test_idx),
        'sample_index': list(train_idx) + list(val_idx) + list(test_idx)
    }
    df = pd.DataFrame(split_data)
    file_path = f"{RESULT_DIR}/split_evidence_{cancer}_{task}_{mode}.csv"
    df.to_csv(file_path, index=False)
    return file_path

def plot_learning_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss, all_preds, all_targets = 0.0, [], []
    for inputs_dict, labels in dataloader:
        inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs_dict)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    return running_loss / len(dataloader.dataset), accuracy_score(all_targets, all_preds)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for inputs_dict, labels in dataloader:
            inputs_dict = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
            labels = labels.to(DEVICE)
            outputs = model(inputs_dict)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, f1, prec, rec

def run_experiment(use_protein):
    set_seed(SEED)
    DATA_ROOT = 'processed_data_common_no_protein'
    mode = "with_protein" if use_protein else "no_protein"
    print(f"\n{'='*25} STARTING EXPERIMENT: {mode.upper()} {'='*25}")

    for cancer in CANCER_LIST:
        for task in TASKS:
            result = B_dataset_loader.load_and_process_data(cancer, task_type=task, use_protein=use_protein, data_root=DATA_ROOT)
            if result is None: continue
            
            full_ds = ConcatDataset([result[0], result[1]])
            total_size = len(full_ds)
            train_size = int(0.7 * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size
            
            train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size], 
                                                     generator=torch.Generator().manual_seed(SEED))
            
            save_split_evidence(train_ds.indices, val_ds.indices, test_ds.indices, cancer, task, mode)
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            model = B_models.MoXGATE(result[2], num_classes=result[3], num_heads=32).to(DEVICE)
            criterion = B_models.FocalLoss() 
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
            
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            best_val_loss, early_stop_counter = float('inf'), 0
            
            for epoch in range(EPOCHS):
                t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
                v_loss, v_acc, _, _, _ = evaluate(model, val_loader, criterion)
                
                history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
                history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
                
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    torch.save(model.state_dict(), f"{RESULT_DIR}/best_{cancer}_{task}_{mode}.pth")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if early_stop_counter >= PATIENCE: break
            
            # Đánh giá tập TEST và in đầy đủ 4 chỉ số
            model.load_state_dict(torch.load(f"{RESULT_DIR}/best_{cancer}_{task}_{mode}.pth"))
            te_loss, te_acc, te_f1, te_prec, te_rec = evaluate(model, test_loader, criterion)
            
            print(f"   [FINAL TEST] {cancer} - {task}:")
            print(f"     -> Accuracy  : {te_acc:.4f}")
            print(f"     -> F1-Score  : {te_f1:.4f}")
            print(f"     -> Precision : {te_prec:.4f}")
            print(f"     -> Recall    : {te_rec:.4f}")
            
            plot_learning_curves(history, f"{RESULT_DIR}/curves_{cancer}_{task}_{mode}.png")

if __name__ == "__main__":
    run_experiment(use_protein=False)
    run_experiment(use_protein=True)