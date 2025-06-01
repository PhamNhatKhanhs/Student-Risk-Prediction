# src/data_loader.py
import pandas as pd
from src import config

def load_dataset(file_path=config.DATA_FILE_PATH) -> pd.DataFrame:
    """Tải dữ liệu từ file CSV."""
    try:
        # Thêm tham số sep=';' vào đây
        df = pd.read_csv(file_path, sep=';')
        print(f"Dữ liệu đã được tải thành công từ: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{file_path}'.")
        print("Vui lòng tải file student-mat.csv và đặt vào thư mục 'data/'.")
        return None
    except Exception as e:
        print(f"Lỗi không xác định khi tải dữ liệu: {e}")
        return None

if __name__ == '__main__':
    # Test thử hàm load_dataset
    data = load_dataset()
    if data is not None:
        print("\n5 dòng đầu của dữ liệu:")
        print(data.head())
        print(f"\nKích thước dữ liệu: {data.shape}")
        print("\nCác cột trong DataFrame là:", data.columns.tolist()) # Để kiểm tra tên cột