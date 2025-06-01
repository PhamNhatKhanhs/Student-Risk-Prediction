# src/predict_utils.py
import pandas as pd

def predict_new_student(pipeline, student_data_dict: dict, original_feature_columns: list):
    """
    Dự đoán nguy cơ cho một học sinh mới.
    student_data_dict: dictionary chứa thông tin học sinh, key là tên cột.
    original_feature_columns: danh sách các cột feature gốc (trước khi one-hot) để đảm bảo thứ tự.
    """
    try:
        # Tạo DataFrame từ dictionary, đảm bảo đúng thứ tự cột như khi train
        # Điều này quan trọng vì preprocessor (đặc biệt là OneHotEncoder) đã học thứ tự này
        student_df = pd.DataFrame([student_data_dict], columns=original_feature_columns)
        
        prediction_label = pipeline.predict(student_df)[0]
        prediction_proba = pipeline.predict_proba(student_df)[0] # Xác suất cho cả 2 lớp
        
        risk_status_str = "Nguy cơ cao" if prediction_label == 1 else "An toàn"
        proba_risk = prediction_proba[1] # Xác suất là Nguy cơ cao (lớp 1)
        
        print(f"\n--- Dự đoán cho học sinh mới ---")
        print(f"Thông tin đầu vào: {student_data_dict}")
        print(f"Dự đoán: {risk_status_str}")
        print(f"Xác suất là 'Nguy cơ cao': {proba_risk:.4f}")
        
        return risk_status_str, proba_risk
        
    except Exception as e:
        print(f"Lỗi khi dự đoán cho học sinh mới: {e}")
        print("Hãy kiểm tra lại dữ liệu đầu vào và danh sách cột feature gốc.")
        return None, None