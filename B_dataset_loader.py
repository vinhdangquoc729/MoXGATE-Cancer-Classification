import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import os
import re

class MultiOmicsDataset(Dataset):
    def __init__(self, data_dict, labels):
        self.data_dict = data_dict
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.modalities = list(data_dict.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {mod: torch.tensor(self.data_dict[mod][idx], dtype=torch.float32) for mod in self.modalities}
        label = self.labels[idx]
        return sample, label

# --- HÀM CHUẨN HÓA ID ---
def normalize_id(patient_id):
    """TCGA.5L.AAT0 -> TCGA-5L-AAT0 (12 ký tự)"""
    s = str(patient_id).upper().strip().replace('.', '-')
    match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', s)
    if match: return match.group(1)
    return s[:12]

# --- HÀM KIỂM TRA XOAY TRỤC ---
def ensure_patients_in_rows(df, file_name=""):
    # Nếu index bắt đầu bằng TCGA -> Đúng
    if df.index.astype(str)[0].strip().upper().startswith('TCGA'):
        return df
    # Nếu cột bắt đầu bằng TCGA -> Ngược -> Xoay
    if df.columns.astype(str)[0].strip().upper().startswith('TCGA'):
        return df.T
    # Fallback: số cột > số dòng nhiều -> Xoay
    if df.shape[1] > df.shape[0]:
        return df.T
    return df

def load_and_process_data(cancer_type, task_type='stage', use_protein=False, data_root='./processed_data_common', label_root='./classify'):
    print(f"\n[{cancer_type}] Đang tải dữ liệu cho bài toán: {task_type}...")
    
    omics_files = {'rna': 'rna.csv', 'meth': 'meth.csv', 'mirna': 'miRNA.csv', 'cn': 'CN.csv'}
    
    if use_protein:
        gen_prot_path = os.path.join(data_root, cancer_type, 'protein_new.fea')
        if os.path.exists(gen_prot_path):
             omics_files['protein'] = 'protein_new.fea'
             print(f"   -> Sử dụng Protein dự đoán: {gen_prot_path}")
        else:
             omics_files['protein'] = 'protein.csv'
             print("   -> Sử dụng Protein Gốc.")

    # --- 1. LOAD LABEL ---
    label_path = os.path.join(label_root, f"{cancer_type}.csv")
    if not os.path.exists(label_path):
        print(f"   [LỖI] Không tìm thấy file nhãn: {label_path}")
        return None

    try:
        try:
            label_df = pd.read_csv(label_path, sep=',', index_col=0)
            if label_df.shape[1] < 2: label_df = pd.read_csv(label_path, sep='\t', index_col=0)
        except: return None

        label_df = ensure_patients_in_rows(label_df, "Label")
        label_df.index = label_df.index.map(normalize_id)
        label_df = label_df[~label_df.index.duplicated(keep='first')]
        
        target_col = get_target_column(label_df, task_type)
        if target_col is None:
            print(f"   [SKIP] Không tìm thấy cột {task_type}")
            return None
            
        label_series = clean_labels(label_df[target_col], task_type)
        label_series = label_series.dropna()
        
    except Exception as e:
        print(f"   [LỖI] Label processing: {e}")
        return None

    # --- 2. LOAD OMICS ---
    loaded_data = {}
    
    for mod, filename in omics_files.items():
        file_path = os.path.join(data_root, cancer_type, filename)
        if not os.path.exists(file_path): continue
        # print(file_path)
        try:
            if filename.endswith('.fea'):
                df = pd.read_csv(file_path, sep='\t', index_col=0)
            else:
                df = pd.read_csv(file_path, index_col=0)
            
            df = ensure_patients_in_rows(df, filename)
            df.index = df.index.map(normalize_id)
            
            df = df.groupby(df.index).mean()
            
            loaded_data[mod] = df
            # print(f"   -> Loaded {mod}: {df.shape}") # Log để kiểm tra
        except Exception as e: 
            print(f"   [LỖI] Không đọc được file {filename}: {e}")
            continue

    if not loaded_data:
        print(f"   [STOP] Không load được bất kỳ file Omics nào!")
        return None

    # --- 3. TÌM GIAO ĐIỂM ---
    common_patients = set(label_series.index)
    for mod, df in loaded_data.items():
        common_patients = common_patients.intersection(set(df.index))

    if len(common_patients) < 30:
        print(f"   [SKIP] Quá ít bệnh nhân chung ({len(common_patients)} mẫu).")
        return None

    print(f"   -> Số bệnh nhân chung: {len(common_patients)}")
    common_patients = sorted(list(common_patients))

    # --- 4. CHUẨN HÓA DỮ LIỆU ---
    final_data_dict = {}
    for mod, df in loaded_data.items():
        df_filtered = df.loc[common_patients].fillna(0)
        scaler = StandardScaler()
        final_data_dict[mod] = scaler.fit_transform(df_filtered.values)

    # --- 5. XỬ LÝ NHÃN (LỌC & RE-ENCODE) ---
    final_labels_raw = label_series.loc[common_patients]
    le_initial = LabelEncoder()
    temp_labels = le_initial.fit_transform(final_labels_raw)
    
    label_counts = Counter(temp_labels)
    valid_classes = [cls for cls, count in label_counts.items() if count >= 2]
    
    if len(valid_classes) < 2:
        print(f"   [SKIP] Không đủ class (Cần ít nhất 2 loại mô học khác nhau).")
        return None
        
    mask = np.isin(temp_labels, valid_classes)
    indices = np.arange(len(common_patients))[mask]
    filtered_labels = temp_labels[mask]
    
    le_final = LabelEncoder()
    final_labels_encoded = le_final.fit_transform(filtered_labels)
    
    # In ra tên class thật để bạn dễ theo dõi
    original_classes = le_initial.inverse_transform(valid_classes)
    print(f"   -> Histology Map: {list(original_classes)} ==> {sorted(list(np.unique(final_labels_encoded)))}")
    
    # --- 6. SPLIT ---
    try:
        train_sub_idx, val_sub_idx = train_test_split(
            np.arange(len(final_labels_encoded)), 
            test_size=0.2, 
            random_state=42, 
            stratify=final_labels_encoded
        )
    except: return None
    
    train_idx = indices[train_sub_idx]
    val_idx = indices[val_sub_idx]

    train_data = {mod: final_data_dict[mod][train_idx] for mod in final_data_dict}
    val_data = {mod: final_data_dict[mod][val_idx] for mod in final_data_dict}
    
    return (MultiOmicsDataset(train_data, final_labels_encoded[train_sub_idx]), 
            MultiOmicsDataset(val_data, final_labels_encoded[val_sub_idx]), 
            {mod: final_data_dict[mod].shape[1] for mod in final_data_dict}, 
            len(np.unique(final_labels_encoded)))

# --- HÀM PHỤ TRỢ ĐÃ CẬP NHẬT ---
def get_target_column(df, task_type):
    cols = [c.lower() for c in df.columns]
    
    if task_type == 'stage': 
        candidates = ['pathologic_stage', 'tumor_stage', 'stage_event_pathologic_stage']
    elif task_type == 'n_stage': 
        candidates = ['pathology_n_stage', 'lymph_node_examined_count'] 
    elif task_type == 'm_stage': 
        candidates = ['pathology_m_stage']
    elif task_type == 'histological_type':
        candidates = ['histological_type', 'histology', 'primary_diagnosis']
    # [BỔ SUNG] Tìm cột nhãn cho PAM50 hoặc Molecular Subtype
    elif task_type == 'pam50' or task_type == 'subtype':
        candidates = ['pam50', 'subtype', 'molecular_subtype', 'molecular']
    else:
        # Nếu task lạ, dùng chính tên task làm candidate để tìm kiếm
        candidates = [task_type.lower()]
        
    for target in candidates:
        for col in df.columns:
            if target in col.lower(): return col
    return None

def clean_labels(series, task_type):
    series = series.astype(str).str.lower().str.strip()
    
    def mapper(val):
        # 1. Xử lý các giá trị được coi là "Trống" hoặc "Không xác định"
        # "na" chính là chữ "NA" của bạn sau khi bị .lower()
        if val in ['nan', 'not reported', 'unknown', 'not applicable', 'null']:
            # Nếu là bài toán PAM50, ta coi NA/Trống là nhóm 'normal' theo ý định của bạn
            # if task_type == 'pam50':
            #     return 'normal'
            # Với các bài toán khác (stage, n_stage...), ta trả về np.nan để dropna() lọc bỏ
            return np.nan
        
        # 2. Với các task khác hoặc các giá trị thiếu thật sự
        # Loại bỏ các giá trị nhiễu (Lưu ý: bỏ 'na' ra khỏi danh sách này đối với task pam50)
        # invalid_vals = []
        
            
        # if val in invalid_vals: 
        #     return np.nan
        
        if task_type == 'stage':
            if 'iv' in val: return 3 
            if 'iii' in val: return 2
            if 'ii' in val: return 1
            if 'i' in val and 'is' not in val: return 0
        elif task_type == 'n_stage':
            if 'n0' in val: return 0
            if 'n1' in val or 'n2' in val or 'n3' in val: return 1 
        elif task_type == 'm_stage':
            if 'm0' in val: return 0
            if 'm1' in val: return 1 
            
        # 4. Xử lý Subtype (GIAC) và PAM50 (BRCA)
        elif task_type == 'pam50' or task_type == 'subtype':
            if 'luma' in val or 'luminal a' in val: return 'luma'
            if 'lumb' in val or 'luminal b' in val: return 'lumb'
            if 'her2' in val: return 'her2'
            if 'basal' in val: return 'basal'
            if 'NA' in val: return 'normal'
            return val # Giữ nguyên CIN, MSI, GS...
            
        elif task_type == 'histological_type':
            return val 
            
        return np.nan
    return series.apply(mapper)