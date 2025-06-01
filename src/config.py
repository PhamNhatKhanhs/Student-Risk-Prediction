# src/config.py

# Đường dẫn dữ liệu
DATA_FILE_PATH = "data/student-mat.csv" # Đảm bảo file này tồn tại

# Cấu hình cho biến mục tiêu
TARGET_COLUMN = 'G3' # Cột điểm cuối kỳ gốc
RISK_THRESHOLD = 10  # Ngưỡng để xác định "nguy cơ cao" (G3 < 10)
NEW_TARGET_COLUMN = 'risk_status' # Tên cột mục tiêu mới (0 hoặc 1)

# Cấu hình cho việc chia dữ liệu
TEST_SET_SIZE = 0.2
RANDOM_SEED = 42 # Để đảm bảo kết quả có thể lặp lại

# (Optional) Đường dẫn lưu mô hình
# MODEL_OUTPUT_PATH = "models/"