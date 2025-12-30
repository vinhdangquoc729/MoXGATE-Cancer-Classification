import pandas as pd
import os
import glob
import re

# --- CẤU HÌNH ---
# Thay dấu chấm '.' bằng đường dẫn tới thư mục chứa các file csv clinical của bạn
FOLDER_PATH = 'classify' 
FILE_EXTENSION = '*.csv' # Hoặc '*.txt' nếu file đuôi txt

def clean_stage(val):
    """Gộp các stage chi tiết về 4 nhóm chính"""
    s = str(val).lower().strip()
    if s in ['nan', 'na', 'not reported', 'unknown']: return 'Unknown'
    
    # Ưu tiên check từ cao xuống thấp để tránh nhầm lẫn
    if 'iv' in s: return 'Stage IV'
    if 'iii' in s: return 'Stage III'
    if 'ii' in s: return 'Stage II'
    if 'i' in s and 'is' not in s: return 'Stage I' # Tránh 'tis'
    if '0' in s or 'is' in s: return 'Stage 0'
    return s # Giữ nguyên nếu không map được

def clean_t(val):
    """Làm sạch T-stage (T1a -> T1)"""
    s = str(val).lower().strip()
    if s in ['nan', 'na']: return 'Unknown'
    if 't1' in s: return 'T1'
    if 't2' in s: return 'T2'
    if 't3' in s: return 'T3'
    if 't4' in s: return 'T4'
    if 'tis' in s: return 'Tis'
    if 't0' in s: return 'T0'
    return s

def clean_n(val):
    """Làm sạch N-stage (N1a -> N1)"""
    s = str(val).lower().strip()
    if s in ['nan', 'na']: return 'Unknown'
    if 'n0' in s: return 'N0'
    if 'n1' in s: return 'N1'
    if 'n2' in s: return 'N2'
    if 'n3' in s: return 'N3'
    if 'nx' in s: return 'NX'
    return s

def clean_m(val):
    """Làm sạch M-stage"""
    s = str(val).lower().strip()
    if s in ['nan', 'na']: return 'Unknown'
    if 'm0' in s: return 'M0'
    if 'm1' in s: return 'M1'
    if 'mx' in s: return 'MX'
    return s

def analyze_file(file_path):
    filename = os.path.basename(file_path)
    print(f"\n{'='*50}")
    print(f"ĐANG XỬ LÝ FILE: {filename}")
    
    try:
        # Đọc file (hỗ trợ cả dấu phẩy và tab)
        try:
            df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
            if df.shape[1] < 2: # Nếu đọc bằng dấu phẩy mà chỉ có 1 cột -> thử tab
                df = pd.read_csv(file_path, sep='\t', index_col=0, low_memory=False)
        except:
            print("Lỗi định dạng file (không phải csv/tsv chuẩn).")
            return

        # Xoay trục nếu dữ liệu đang nằm ngang (số cột > số dòng)
        if df.shape[1] > df.shape[0]:
            # print(" -> Phát hiện dữ liệu ngang, đang xoay trục...")
            df = df.T

        # Chuẩn hóa tên cột về chữ thường để dễ tìm
        df.columns = df.columns.astype(str).str.lower().str.strip()
        
        # --- 1. THỐNG KÊ STAGE ---
        stage_col = next((c for c in df.columns if 'pathologic_stage' in c or 'tumor_stage' in c), None)
        if stage_col:
            print(f"\n--- GIAI ĐOẠN (Overall Stage) ---")
            counts = df[stage_col].apply(clean_stage).value_counts().sort_index()
            for k, v in counts.items():
                print(f"  {k}: {v}")
        else:
            print("\n[!] Không tìm thấy cột Pathologic Stage")

        # --- 2. THỐNG KÊ T - N - M ---
        # T Stage
        t_col = next((c for c in df.columns if 'pathology_t_stage' in c), None)
        if t_col:
            print(f"\n--- T STAGE (Tumor) ---")
            print(df[t_col].apply(clean_t).value_counts().to_string())
        else:
            print("\n[!] Không tìm thấy cột T Stage")

        # N Stage
        n_col = next((c for c in df.columns if 'pathology_n_stage' in c), None)
        if n_col:
            print(f"\n--- N STAGE (Lymph Node) ---")
            print(df[n_col].apply(clean_n).value_counts().to_string())
        else:
            print("\n[!] Không tìm thấy cột N Stage")

        # M Stage
        m_col = next((c for c in df.columns if 'pathology_m_stage' in c), None)
        if m_col:
            print(f"\n--- M STAGE (Metastasis) ---")
            print(df[m_col].apply(clean_m).value_counts().to_string())
        else:
            print("\n[!] Không tìm thấy cột M Stage")

        # --- 3. THỐNG KÊ HISTOLOGICAL TYPE ---
        # Tìm cột histological type (có thể tên khác nhau chút)
        hist_col = next((c for c in df.columns if 'histological_type' in c or 'histology' in c), None)
        if hist_col:
            print(f"\n--- HISTOLOGICAL TYPE (Loại mô học) ---")
            # Lấy top 10 loại phổ biến nhất nếu quá nhiều
            counts = df[hist_col].value_counts()
            if len(counts) > 0:
                print(counts.to_string())
            else:
                print("  (Trống)")
        else:
            print("\n[!] Không tìm thấy cột Histological Type")

    except Exception as e:
        print(f"Lỗi khi xử lý file {filename}: {e}")

# --- CHẠY ---
if __name__ == "__main__":
    files = glob.glob(os.path.join(FOLDER_PATH, FILE_EXTENSION))
    if not files:
        print("Không tìm thấy file .csv nào trong thư mục!")
    else:
        print(f"Tìm thấy {len(files)} file. Bắt đầu thống kê...")
        for f in files:
            analyze_file(f)