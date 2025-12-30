import os
import pandas as pd

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = '.' 
DIR_FEA_ROOT = os.path.join(BASE_DIR, 'fea')
DIR_PROTEIN_ROOT = os.path.join(BASE_DIR, 'protein')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data_common_no_protein')
OUTPUT_DIR_ALL = os.path.join(OUTPUT_DIR, 'all') # Thư mục chứa file gộp

# Danh sách folder
FOLDERS = ['BLCA', 'BRCA', 'KIRC', 'LUAD', 'SKCM', 'STAD', 'UCEC', 'UVM']

# Các loại dữ liệu
DATA_TYPES = {
    'CN': DIR_FEA_ROOT,
    'meth': DIR_FEA_ROOT,
    'miRNA': DIR_FEA_ROOT,
    'rna': DIR_FEA_ROOT,
    # 'protein': DIR_PROTEIN_ROOT
}

def get_file_path(folder_name, data_type):
    """Hàm lấy đường dẫn file gốc"""
    root = DATA_TYPES[data_type]
    filename = f"{data_type}.fea"
    return os.path.join(root, folder_name, filename)

def clean_string(s):
    """Làm sạch chuỗi (bỏ ngoặc kép, khoảng trắng)"""
    return str(s).replace('"', '').replace("'", "").strip()

def step_1_find_global_common_features():
    print("=== BƯỚC 1: TÌM FEATURE CHUNG TOÀN CỤC ===")
    common_features = {dtype: None for dtype in DATA_TYPES}

    for dtype in DATA_TYPES:
        print(f"-> Quét: {dtype}...")
        for folder in FOLDERS:
            fpath = get_file_path(folder, dtype)
            if not os.path.exists(fpath):
                continue
            try:
                # Chỉ đọc cột đầu tiên
                df_idx = pd.read_csv(fpath, usecols=[0])
                current_feats = set(df_idx.iloc[:, 0].apply(clean_string))
                
                if common_features[dtype] is None:
                    common_features[dtype] = current_feats
                else:
                    common_features[dtype] = common_features[dtype].intersection(current_feats)
            except Exception as e:
                print(f"   Lỗi {fpath}: {e}")
    return common_features

def step_2_process_individual_folders(global_common_features):
    print("\n=== BƯỚC 2: LỌC BỆNH NHÂN CHUNG & LƯU TỪNG FOLDER ===")
    
    # Biến này để lưu lại danh sách folder đã xử lý thành công -> Dùng cho bước 3
    processed_folders = []

    for folder in FOLDERS:
        print(f"\nProcessing: {folder}")
        
        # 2a. Tìm bệnh nhân chung trong nội bộ folder
        common_patients = None
        valid_folder = True
        
        for dtype in DATA_TYPES:
            fpath = get_file_path(folder, dtype)
            if not os.path.exists(fpath):
                valid_folder = False; break
            try:
                df_header = pd.read_csv(fpath, nrows=0)
                patients = df_header.columns[1:] 
                cleaned_patients = set([clean_string(p) for p in patients])
                
                if common_patients is None:
                    common_patients = cleaned_patients
                else:
                    common_patients = common_patients.intersection(cleaned_patients)
            except:
                valid_folder = False; break
        
        if not valid_folder or not common_patients:
            print(f"   => Bỏ qua (Thiếu file hoặc không có bệnh nhân chung)")
            continue
        print(f"Số bệnh nhân chung: {len(common_patients)}")
        # Lưu danh sách folder thành công
        processed_folders.append(folder)
        
        # 2b. Lưu file đã lọc
        save_folder = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(save_folder, exist_ok=True)
        
        sorted_patients = sorted(list(common_patients))
        
        for dtype in DATA_TYPES:
            fpath = get_file_path(folder, dtype)
            df = pd.read_csv(fpath)
            
            # Chuẩn hóa Index
            df.iloc[:, 0] = df.iloc[:, 0].apply(clean_string)
            df = df.set_index(df.columns[0])
            
            # Chuẩn hóa Columns
            df.columns = [clean_string(col) for col in df.columns]
            
            # Lọc
            try:
                feats_to_keep = global_common_features[dtype]
                df_filtered = df.loc[df.index.isin(feats_to_keep), sorted_patients]
                
                # Sắp xếp lại Index (Feature) để đảm bảo đồng bộ cho Bước 3
                df_filtered = df_filtered.sort_index()
                
                # Reset index để lưu cột tên feature vào file
                df_filtered.reset_index(inplace=True)
                
                save_path = os.path.join(save_folder, f"{dtype}.csv")
                df_filtered.to_csv(save_path, index=False)
            except Exception as e:
                print(f"   Lỗi lưu {dtype}: {e}")
    
    return processed_folders

def step_3_merge_all_folders(processed_folders):
    print("\n=== BƯỚC 3: GỘP TẤT CẢ VÀO THƯ MỤC 'ALL' ===")
    
    if not processed_folders:
        print("Không có folder nào được xử lý thành công để gộp.")
        return

    os.makedirs(OUTPUT_DIR_ALL, exist_ok=True)

    for dtype in DATA_TYPES:
        print(f"-> Đang gộp dữ liệu: {dtype}...")
        dfs_to_merge = []
        
        for folder in processed_folders:
            # Đọc lại file CSV đã xử lý sạch ở Bước 2
            file_path = os.path.join(OUTPUT_DIR, folder, f"{dtype}.csv")
            
            if os.path.exists(file_path):
                # index_col=0 để lấy cột feature làm index -> tiện cho việc nối cột (axis=1)
                df = pd.read_csv(file_path, index_col=0)
                dfs_to_merge.append(df)
            else:
                print(f"   Cảnh báo: Không tìm thấy file đã xử lý của {folder}")

        if dfs_to_merge:
            try:
                # Nối các dataframe theo chiều ngang (axis=1)
                # Vì Bước 2 đã sort_index(), các dòng sẽ tự động khớp nhau
                merged_df = pd.concat(dfs_to_merge, axis=1)
                
                # Reset index để đưa cột Feature ra ngoài cùng khi lưu
                merged_df.reset_index(inplace=True)
                
                save_path = os.path.join(OUTPUT_DIR_ALL, f"{dtype}.csv")
                merged_df.to_csv(save_path, index=False)
                
                print(f"   => Đã xong {dtype}. Kích thước: {merged_df.shape}")
            except Exception as e:
                print(f"   Lỗi khi gộp {dtype}: {e}")

if __name__ == "__main__":
    # 1. Tìm features chung của 8 bộ
    global_feats = step_1_find_global_common_features()
    
    # 2. Xử lý riêng từng folder (Lọc feature chung & bệnh nhân chung của bộ đó)
    success_folders = step_2_process_individual_folders(global_feats)
    
    # 3. Gộp 8 folder lại thành 1 file to cho mỗi loại dữ liệu
    step_3_merge_all_folders(success_folders)
    
    print("\n=== HOÀN TẤT TOÀN BỘ ===")
    print(f"Dữ liệu gộp nằm tại: {OUTPUT_DIR_ALL}")