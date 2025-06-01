# app.py
import streamlit as st
import pandas as pd
import joblib
import io # Cần thiết để tạo file CSV tải xuống trong bộ nhớ

# --- LỆNH STREAMLIT ĐẦU TIÊN ---
st.set_page_config(
    page_title="Dự đoán Nguy cơ Học tập",
    page_icon="🎓",
    layout="wide"
)

# --- Cấu hình các lựa chọn cho categorical features ---
categorical_options = {
    'school': ['GP', 'MS'], 'sex': ['F', 'M'], 'address': ['U', 'R'], 
    'famsize': ['LE3', 'GT3'], 'Pstatus': ['T', 'A'], 
    'Mjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'Fjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'reason': ['home', 'reputation', 'course', 'other'],
    'guardian': ['mother', 'father', 'other'],
    'schoolsup': ['yes', 'no'], 'famsup': ['yes', 'no'], 'paid': ['yes', 'no'], 
    'activities': ['yes', 'no'], 'nursery': ['yes', 'no'], 'higher': ['yes', 'no'], 
    'internet': ['yes', 'no'], 'romantic': ['yes', 'no']
}

# --- Tải mô hình và danh sách cột ---
MODEL_FILENAME = 'random_forest_pipeline.joblib' 
FEATURE_COLUMNS_FILENAME = 'feature_columns.joblib'

@st.cache_resource 
def load_model_and_cols():
    try:
        pipeline = joblib.load(MODEL_FILENAME)
        feature_cols = joblib.load(FEATURE_COLUMNS_FILENAME)
        return pipeline, feature_cols
    except FileNotFoundError:
        return None, None
    except Exception as e: 
        st.error(f"Lỗi khi tải mô hình hoặc file cột: {e}")
        return None, None

pipeline, feature_columns = load_model_and_cols()

# --- Tiêu đề và Giới thiệu ---
st.title("🎓 Hệ thống Dự đoán Nguy cơ Học tập của Học sinh")
st.markdown("""
Chào mừng! Công cụ này dùng Machine Learning (Random Forest) để ước tính khả năng học sinh đạt kết quả thấp (G3 < 10).
Bạn có thể nhập thông tin cho từng học sinh hoặc tải lên file CSV để dự đoán hàng loạt.
""")
st.markdown("---")

if pipeline is None or feature_columns is None:
    st.error(f"Lỗi: Không tìm thấy file '{MODEL_FILENAME}' hoặc '{FEATURE_COLUMNS_FILENAME}'.")
    st.warning("Vui lòng chạy `python main.py` để huấn luyện và lưu mô hình trước.")
    st.stop()

# --- Tab cho nhập liệu thủ công và tải file ---
tab1, tab2 = st.tabs(["📝 Nhập liệu Thủ công", "📂 Tải lên File CSV"])

with tab1:
    st.subheader("Nhập thông tin cho một học sinh:")
    with st.form(key="student_info_form_manual"):
        input_data_manual = {}
        with st.expander("💡 Hướng dẫn về các yếu tố đầu vào (Nhấn để xem)", expanded=False):
            st.markdown("""
            * **G1, G2:** Điểm giữa kỳ 1 và giữa kỳ 2 (0-20).
            * **absences:** Số buổi vắng học.
            * **studytime:** Thời gian học hàng tuần (1: <2 giờ, 2: 2-5 giờ, 3: 5-10 giờ, 4: >10 giờ).
            * **failures:** Số lần thi trượt các môn trước đó (0-4).
            * *(Và các thông tin khác về trường, gia đình, thói quen...)*
            """)

        col1_manual, col2_manual, col3_manual = st.columns(3)
        cols_for_layout_manual = [col1_manual, col2_manual, col3_manual]
        
        for i, feature in enumerate(feature_columns):
            target_col_manual = cols_for_layout_manual[i % 3]
            help_text = None; caption_text = None
            
            if feature == 'Medu': caption_text = "Học vấn mẹ (0-4)"
            elif feature == 'Fedu': caption_text = "Học vấn cha (0-4)"
            elif feature == 'studytime': help_text = "1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h / tuần"
            elif feature == 'failures': help_text = "Số lần thi trượt trước (0-4)"
            elif feature in ['G1', 'G2']: help_text = "Điểm (0-20)"

            if feature in categorical_options:
                input_data_manual[feature] = target_col_manual.selectbox(f"Chọn {feature}:", options=categorical_options[feature], key=f"manual_{feature}", help=help_text)
            elif feature in ['G1', 'G2', 'age', 'absences'] or feature in ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']:
                default_val_map = {'age': 16, 'absences': 0, 'G1':10, 'G2':10, 'Medu':2, 'Fedu':2, 'traveltime':1, 'studytime':2, 'failures':0, 'famrel':3, 'freetime':3, 'goout':3, 'Dalc':1, 'Walc':1, 'health':3}
                min_val_map = {'age': 15, 'absences': 0, 'G1':0, 'G2':0, 'Medu':0, 'Fedu':0, 'traveltime':1, 'studytime':1, 'failures':0, 'famrel':1, 'freetime':1, 'goout':1, 'Dalc':1, 'Walc':1, 'health':1}
                max_val_map = {'age': 22, 'absences': 93, 'G1':20, 'G2':20, 'Medu':4, 'Fedu':4, 'traveltime':4, 'studytime':4, 'failures':4, 'famrel':5, 'freetime':5, 'goout':5, 'Dalc':5, 'Walc':5, 'health':5}
                
                input_data_manual[feature] = target_col_manual.number_input(
                    f"Nhập {feature}:", 
                    min_value=min_val_map.get(feature,0), 
                    max_value=max_val_map.get(feature,20 if feature in ['G1','G2'] else (93 if feature=='absences' else (22 if feature=='age' else 5 ) ) ), # Điều chỉnh max value
                    value=default_val_map.get(feature,0), 
                    step=1, key=f"manual_{feature}", help=help_text
                )
            else: 
                input_data_manual[feature] = target_col_manual.number_input(f"Nhập {feature} (khác):", value=0, key=f"manual_{feature}", help=help_text)
            
            if caption_text: target_col_manual.caption(caption_text)

        submit_button_manual = st.form_submit_button(label="🚀 Dự đoán Nguy cơ (Thủ công)", use_container_width=True, type="primary")

    if submit_button_manual:
        ordered_input_data_manual = {col: input_data_manual[col] for col in feature_columns}
        input_df_manual = pd.DataFrame([ordered_input_data_manual])
        st.markdown("---")
        st.subheader("📈 Kết quả Dự đoán (Thủ công):")
        try:
            prediction_label = pipeline.predict(input_df_manual)[0]
            prediction_proba = pipeline.predict_proba(input_df_manual)[0]
            risk_status_str = "⚠️ Nguy cơ cao (G3 < 10)" if prediction_label == 1 else "✅ An toàn (G3 >= 10)"
            proba_risk = prediction_proba[1]
            res_col1_manual, res_col2_manual = st.columns([2,3])
            with res_col1_manual:
                if prediction_label == 1: st.error(f"**Trạng thái: {risk_status_str}**")
                else: st.success(f"**Trạng thái: {risk_status_str}**")
            with res_col2_manual:
                st.metric(label="Xác suất là 'Nguy cơ cao'", value=f"{proba_risk:.2%}")
                st.progress(proba_risk)
            
            st.markdown("---")
            st.subheader("📝 Lời khuyên:")
            if proba_risk > 0.7: st.warning("Học sinh này có nguy cơ **rất cao**. Cần quan tâm đặc biệt và hỗ trợ cụ thể.")
            elif proba_risk > 0.4: st.info("Học sinh này có **dấu hiệu nguy cơ**. Nên theo dõi và khuyến khích thêm.")
            else: st.balloons(); st.info("Học sinh này có vẻ học tập tốt. Tiếp tục duy trì và khuyến khích!")
        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")

with tab2:
    st.subheader("Tải lên file CSV để dự đoán hàng loạt:")
    st.markdown("File CSV cần có các cột giống như dữ liệu huấn luyện và được phân tách bằng dấu chấm phẩy (`;`).")
    
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"], help="File phải có các cột: " + ", ".join(feature_columns[:5]) + "...") # Hiển thị 5 cột đầu làm ví dụ

    if uploaded_file is not None:
        try:
            # Quan trọng: Đảm bảo đọc file với đúng separator
            new_data_df = pd.read_csv(uploaded_file, sep=';')
            st.success("Đã tải lên và đọc file CSV thành công!")
            st.write("Xem trước 5 dòng dữ liệu từ file đã tải:", new_data_df.head())

            # --- KIỂM TRA VÀ CĂN CHỈNH CỘT ---
            # 1. Kiểm tra các cột bị thiếu so với feature_columns
            missing_cols = set(feature_columns) - set(new_data_df.columns)
            if missing_cols:
                st.error(f"Lỗi: File CSV tải lên bị thiếu các cột bắt buộc sau: `{', '.join(missing_cols)}`")
                st.error(f"Mô hình cần các cột: `{', '.join(feature_columns)}`")
                st.stop() # Dừng xử lý nếu thiếu cột

            # 2. Chọn đúng các cột theo thứ tự của feature_columns, bỏ qua các cột thừa
            # Điều này cực kỳ quan trọng để pipeline tiền xử lý hoạt động đúng
            try:
                df_to_predict = new_data_df[feature_columns].copy()
            except KeyError as e:
                st.error(f"Lỗi: Không thể truy cập một hoặc nhiều cột cần thiết từ file CSV. Đảm bảo tên cột trong file CSV của bạn khớp chính xác (phân biệt chữ hoa/thường) với các cột mà mô hình mong đợi. Lỗi chi tiết: {e}")
                st.error(f"Các cột mô hình mong đợi: `{', '.join(feature_columns)}`")
                st.error(f"Các cột có trong file bạn tải lên: `{', '.join(new_data_df.columns.tolist())}`")
                st.stop()
            
            st.write("Dữ liệu sau khi chọn và sắp xếp cột (5 dòng đầu):", df_to_predict.head())

            if st.button("📊 Thực hiện Dự đoán cho Toàn bộ File CSV", use_container_width=True, type="primary"):
                with st.spinner("⏳ Đang xử lý và dự đoán... Vui lòng đợi."):
                    predictions_batch = pipeline.predict(df_to_predict)
                    probabilities_batch = pipeline.predict_proba(df_to_predict)

                    # Tạo DataFrame kết quả
                    results_df_batch = df_to_predict.copy() 
                    results_df_batch['Dự đoán Nguy cơ (Nhãn)'] = ["Nguy cơ cao" if p == 1 else "An toàn" for p in predictions_batch]
                    results_df_batch['Xác suất Nguy cơ cao'] = [p[1] for p in probabilities_batch] # Lưu dạng số để dễ sort/filter
                
                st.success("🎉 Hoàn thành dự đoán cho toàn bộ file!")
                st.subheader("Kết quả Dự đoán Hàng loạt:")
                st.dataframe(results_df_batch)

                # Cho phép tải xuống kết quả
                # Sử dụng io.StringIO để tạo file CSV trong bộ nhớ
                csv_buffer = io.StringIO()
                results_df_batch.to_csv(csv_buffer, sep=';', index=False, encoding='utf-8-sig') # utf-8-sig để Excel đọc tiếng Việt tốt
                
                st.download_button(
                    label="📥 Tải xuống Kết quả Dự đoán (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="predictions_student_risk_batch.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except pd.errors.ParserError:
            st.error("Lỗi đọc file CSV: Có vẻ file không được phân tách bằng dấu chấm phẩy (';') hoặc có lỗi định dạng khác. Vui lòng kiểm tra lại file.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi không mong muốn khi xử lý file: {e}")


# --- Footer ---
st.markdown("---")
st.caption("Hackathon AI - 01/06/2025 | Dự án: Dự đoán Nguy cơ Học tập của Học sinh")