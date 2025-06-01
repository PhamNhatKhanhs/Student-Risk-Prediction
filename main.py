# main.py
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_dataset
from src.preprocessing import create_target_variable, get_feature_and_target, get_feature_types
from src.model_trainer import train_and_evaluate_models
from src.predict_utils import predict_new_student # (Tùy chọn)
import joblib

def run_pipeline():
    """Chạy toàn bộ pipeline: tải dữ liệu, tiền xử lý, huấn luyện, đánh giá."""
    
    print("===== BẮT ĐẦU PIPELINE DỰ ĐOÁN NGUY CƠ HỌC TẬP =====")
    
    # 1. Tải dữ liệu
    df_raw = load_dataset()
    if df_raw is None:
        return # Dừng nếu không tải được dữ liệu

    # 2. Tiền xử lý
    df_processed = create_target_variable(df_raw)
    X, y = get_feature_and_target(df_processed)
    numerical_cols, categorical_cols = get_feature_types(X)
    
    # Ghi lại tên các cột features gốc để dùng cho predict_new_student
    original_feature_columns_for_prediction = X.columns.tolist()


    # 3. Chia dữ liệu Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SET_SIZE, 
        random_state=config.RANDOM_SEED, 
        stratify=y # Giữ tỷ lệ các lớp trong cả tập train và test
    )
    print(f"\nĐã chia dữ liệu: Train ({X_train.shape}), Test ({X_test.shape})")

    # 4. Huấn luyện và Đánh giá các mô hình
    trained_models, best_model_pipeline, best_model_name = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, 
        numerical_cols, categorical_cols
    )
    
     # 5. Lưu mô hình tốt nhất và danh sách cột (THÊM PHẦN NÀY)
    if best_model_pipeline and best_model_name:
        model_filename = f'{best_model_name.replace(" ", "_").lower()}_pipeline.joblib'
        joblib.dump(best_model_pipeline, model_filename)
        print(f"\nĐã lưu pipeline tốt nhất ({best_model_name}) vào file: {model_filename}")
        
        # Lưu cả danh sách tên cột features gốc
        feature_columns_filename = 'feature_columns.joblib'
        joblib.dump(original_feature_columns_for_prediction, feature_columns_filename)
        print(f"Đã lưu danh sách cột feature vào file: {feature_columns_filename}")

    # 6.Demo dự đoán với mô hình tốt nhất
    if best_model_pipeline:
        print(f"\n===== DEMO DỰ ĐOÁN VỚI MÔ HÌNH TỐT NHẤT: {best_model_name} =====")
        sample_student_dict = {
            'school': 'GP', 'sex': 'F', 'age': 18, 'address': 'U', 'famsize': 'GT3', 
            'Pstatus': 'A', 'Medu': 4, 'Fedu': 4, 'Mjob': 'at_home', 'Fjob': 'teacher', 
            'reason': 'course', 'guardian': 'mother', 'traveltime': 2, 'studytime': 2, 
            'failures': 0, 'schoolsup': 'yes', 'famsup': 'no', 'paid': 'no', 
            'activities': 'no', 'nursery': 'yes', 'higher': 'yes', 'internet': 'no', 
            'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 1, 
            'Walc': 1, 'health': 3, 'absences': 4, 
            'G1': 10, 'G2': 11
        }
        predict_new_student(best_model_pipeline, sample_student_dict, original_feature_columns_for_prediction)

    print("\n===== PIPELINE HOÀN THÀNH =====")

if __name__ == '__main__':
    run_pipeline()