# app.py
import streamlit as st
import pandas as pd
import joblib
from src import config # Để lấy các giá trị mặc định nếu cần

# --- LỆNH STREAMLIT ĐẦU TIÊN ---
st.set_page_config(page_title="Dự đoán Nguy cơ Học tập", layout="wide")

# --- Cấu hình các lựa chọn cho categorical features (quan trọng!) ---
# Dựa trên kiến thức về bộ dữ liệu student-mat.csv
# ... (giữ nguyên phần categorical_options) ...
categorical_options = {
    'school': ['GP', 'MS'],
    'sex': ['F', 'M'],
    'address': ['U', 'R'], # Urban, Rural
    'famsize': ['LE3', 'GT3'], # Less or equal to 3, Greater than 3
    'Pstatus': ['T', 'A'], # Together, Apart
    'Mjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'Fjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'reason': ['home', 'reputation', 'course', 'other'],
    'guardian': ['mother', 'father', 'other'],
    'schoolsup': ['yes', 'no'],
    'famsup': ['yes', 'no'],
    'paid': ['yes', 'no'], # Extra paid classes within the course subject
    'activities': ['yes', 'no'], # Extra-curricular activities
    'nursery': ['yes', 'no'], # Attended nursery school
    'higher': ['yes', 'no'], # Wants to take higher education
    'internet': ['yes', 'no'], # Internet access at home
    'romantic': ['yes', 'no'] # With a romantic relationship
}

# --- Tải mô hình và danh sách cột ---
MODEL_FILENAME = 'random_forest_pipeline.joblib' # Cập nhật tên file nếu mô hình tốt nhất của bạn khác
FEATURE_COLUMNS_FILENAME = 'feature_columns.joblib'

@st.cache_resource # Cache để không tải lại mô hình mỗi lần tương tác
def load_model_and_cols():
    try:
        pipeline = joblib.load(MODEL_FILENAME)
        feature_cols = joblib.load(FEATURE_COLUMNS_FILENAME)
        return pipeline, feature_cols
    except FileNotFoundError:
        return None, None

pipeline, feature_columns = load_model_and_cols()

# --- Xây dựng giao diện (Phần còn lại) ---
st.title("👨‍🎓 Hệ thống Dự đoán Nguy cơ Học tập của Học sinh")
st.markdown("Nhập thông tin của học sinh để dự đoán nguy cơ học tập (điểm cuối kỳ G3 < 10).")

if pipeline is None or feature_columns is None:
    st.error(f"Lỗi: Không tìm thấy file mô hình '{MODEL_FILENAME}' hoặc '{FEATURE_COLUMNS_FILENAME}'.")
    st.error("Vui lòng chạy `python main.py` trước để huấn luyện và lưu mô hình.")
else:
    # ... (Phần còn lại của code tạo input fields và nút dự đoán giữ nguyên) ...
    st.sidebar.header("Thông tin Học sinh")
    
    input_data = {}

    col1, col2 = st.columns(2)
    current_col_idx = 0

    for feature in feature_columns:
        target_col = col1 if current_col_idx % 2 == 0 else col2
        current_col_idx += 1

        if feature in categorical_options:
            input_data[feature] = target_col.selectbox(
                f"Chọn {feature}:", 
                options=categorical_options[feature],
                key=feature
            )
        elif feature in ['G1', 'G2']:
             input_data[feature] = target_col.number_input(
                f"Nhập điểm {feature} (0-20):", 
                min_value=0, max_value=20, value=10, step=1, key=feature
            )
        elif feature == 'age':
            input_data[feature] = target_col.number_input(
                f"Nhập {feature} (tuổi):", 
                min_value=15, max_value=22, value=16, step=1, key=feature
            )
        elif feature == 'absences':
             input_data[feature] = target_col.number_input(
                f"Nhập {feature} (số buổi vắng):", 
                min_value=0, max_value=93, value=0, step=1, key=feature
            )
        elif feature in ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']:
            min_val, max_val, default_val = 1, 5, 3
            if feature == 'failures': max_val, default_val = 4, 0
            if feature == 'traveltime': max_val, default_val = 4, 1
            if feature == 'studytime': max_val, default_val = 4, 2
            
            input_data[feature] = target_col.number_input(
                f"Nhập {feature} (thang điểm tùy thuộc đặc trưng):",
                min_value=min_val if feature != 'failures' else 0,
                max_value=max_val, 
                value=default_val, 
                step=1, key=feature
            )
        else: 
            input_data[feature] = target_col.text_input(f"Nhập {feature}:", key=feature)


    if st.button("🚀 Dự đoán Nguy cơ", use_container_width=True, type="primary"):
        ordered_input_data = {col: input_data[col] for col in feature_columns}
        input_df = pd.DataFrame([ordered_input_data])

        try:
            prediction_label = pipeline.predict(input_df)[0]
            prediction_proba = pipeline.predict_proba(input_df)[0] 

            risk_status_str = "⚠️ Nguy cơ cao" if prediction_label == 1 else "✅ An toàn"
            proba_risk = prediction_proba[1] 
            
            st.markdown("---")
            st.subheader("Kết quả Dự đoán:")
            
            if prediction_label == 1:
                st.error(f"**Trạng thái: {risk_status_str}**")
            else:
                st.success(f"**Trạng thái: {risk_status_str}**")
            
            st.write(f"**Xác suất là 'Nguy cơ cao':** `{proba_risk:.2%}`")
            
            st.progress(proba_risk)

            if proba_risk > 0.7:
                st.warning("Lời khuyên: Học sinh này có nguy cơ rất cao, cần có sự quan tâm và hỗ trợ đặc biệt.")
            elif proba_risk > 0.4:
                 st.info("Lời khuyên: Học sinh này có dấu hiệu nguy cơ, cần theo dõi và khuyến khích thêm.")
            else:
                st.balloons()

        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {e}")
            st.error("Vui lòng kiểm tra lại các giá trị đầu vào.")

st.sidebar.markdown("---")
st.sidebar.markdown("Hackathon AI - 01/06/2025")
st.sidebar.markdown("Dự án: Dự đoán Nguy cơ Học tập")