# app.py
import streamlit as st
import pandas as pd
import joblib
# from src import config # Có thể không cần nếu không dùng giá trị mặc định từ config ở đây

# --- LỆNH STREAMLIT ĐẦU TIÊN ---
st.set_page_config(
    page_title="Dự đoán Nguy cơ Học tập", 
    page_icon="🎓", # Thêm icon cho tab
    layout="wide"
)

# --- Cấu hình các lựa chọn cho categorical features ---
categorical_options = {
    'school': ['GP', 'MS'],
    'sex': ['F', 'M'],
    'address': ['U', 'R'], 
    'famsize': ['LE3', 'GT3'], 
    'Pstatus': ['T', 'A'], 
    'Mjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'Fjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'reason': ['home', 'reputation', 'course', 'other'],
    'guardian': ['mother', 'father', 'other'],
    'schoolsup': ['yes', 'no'],
    'famsup': ['yes', 'no'],
    'paid': ['yes', 'no'], 
    'activities': ['yes', 'no'], 
    'nursery': ['yes', 'no'], 
    'higher': ['yes', 'no'], 
    'internet': ['yes', 'no'], 
    'romantic': ['yes', 'no']
}

# --- Tải mô hình và danh sách cột ---
MODEL_FILENAME = 'random_forest_pipeline.joblib' 
FEATURE_COLUMNS_FILENAME = 'feature_columns.joblib'

@st.cache_resource # Cache để không tải lại mô hình mỗi lần tương tác
def load_model_and_cols():
    try:
        pipeline = joblib.load(MODEL_FILENAME)
        feature_cols = joblib.load(FEATURE_COLUMNS_FILENAME)
        return pipeline, feature_cols
    except FileNotFoundError:
        return None, None
    except Exception as e: # Bắt các lỗi khác có thể xảy ra khi tải file
        st.error(f"Lỗi khi tải mô hình hoặc file cột: {e}")
        return None, None


pipeline, feature_columns = load_model_and_cols()

# --- Tiêu đề và Giới thiệu ---
st.title("🎓 Hệ thống Dự đoán Nguy cơ Học tập của Học sinh")
st.markdown("""
Chào mừng bạn đến với hệ thống dự đoán nguy cơ học tập! 
Công cụ này sử dụng mô hình Machine Learning (Random Forest) để ước tính khả năng một học sinh có thể đạt kết quả học tập thấp (điểm cuối kỳ môn Toán G3 < 10) dựa trên các thông tin đầu vào.
""")
st.markdown("---")


if pipeline is None or feature_columns is None:
    st.error(f"Lỗi: Không tìm thấy file mô hình huấn luyện ('{MODEL_FILENAME}') hoặc file danh sách cột ('{FEATURE_COLUMNS_FILENAME}').")
    st.warning("Vui lòng chạy `python main.py` trong thư mục dự án để huấn luyện và lưu mô hình trước khi chạy ứng dụng này.")
    st.stop() # Dừng ứng dụng nếu không tải được mô hình

# --- Phần nhập liệu ---
with st.form(key="student_info_form"):
    st.subheader("📝 Vui lòng nhập thông tin của học sinh:")
    
    input_data = {}
    
    # Sử dụng st.expander để giải thích các feature nếu cần
    with st.expander("💡 Hướng dẫn về các yếu tố đầu vào (Nhấn để xem)", expanded=False):
        st.markdown("""
        * **school**: Trường học (GP: Gabriel Pereira, MS: Mousinho da Silveira)
        * **sex**: Giới tính (F: Nữ, M: Nam)
        * **age**: Tuổi (15-22)
        * **address**: Khu vực sống (U: Đô thị, R: Nông thôn)
        * **famsize**: Quy mô gia đình (LE3: <=3 người, GT3: >3 người)
        * **Pstatus**: Tình trạng sống chung của cha mẹ (T: Sống cùng, A: Sống riêng)
        * **Medu, Fedu**: Học vấn của mẹ/cha (0: không, 1: tiểu học, 2: THCS, 3: THPT, 4: sau ĐH)
        * **Mjob, Fjob**: Nghề nghiệp của mẹ/cha
        * **reason**: Lý do chọn trường
        * **guardian**: Người giám hộ
        * **traveltime**: Thời gian di chuyển đến trường (1: <15ph, 2: 15-30ph, 3: 30ph-1h, 4: >1h)
        * **studytime**: Thời gian học hàng tuần (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)
        * **failures**: Số lần thi trượt các môn trước (0-4)
        * **schoolsup, famsup, paid, activities, nursery, higher, internet, romantic**: Hỗ trợ từ trường/gia đình, lớp học thêm, hoạt động ngoại khóa, mẫu giáo, muốn học cao hơn, internet, mối quan hệ tình cảm (yes/no)
        * **famrel, freetime, goout, Dalc, Walc, health**: Quan hệ gia đình, thời gian rảnh, đi chơi, uống rượu ngày thường/cuối tuần, sức khỏe (thang 1-5, tệ đến tốt)
        * **absences**: Số buổi vắng học (0-93)
        * **G1, G2**: Điểm giữa kỳ 1 và giữa kỳ 2 (0-20)
        """)

    col1, col2, col3 = st.columns(3) # Chia thành 3 cột cho thoáng hơn

    # Sắp xếp input fields vào các cột
    cols_for_layout = [col1, col2, col3]
    
    for i, feature in enumerate(feature_columns):
        target_col = cols_for_layout[i % 3] # Chia đều vào 3 cột

        # Tạo tooltips/captions cho một số feature quan trọng hoặc khó hiểu
        help_text = None
        caption_text = None
        
        if feature == 'Medu': caption_text = "Học vấn mẹ (0-4)"
        elif feature == 'Fedu': caption_text = "Học vấn cha (0-4)"
        elif feature == 'studytime': help_text = "1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h / tuần"
        elif feature == 'failures': help_text = "Số lần thi trượt các môn học trước (0-4)"
        elif feature == 'absences': help_text = "Tổng số buổi vắng mặt"
        elif feature == 'G1': help_text = "Điểm giữa kỳ 1 (0-20)"
        elif feature == 'G2': help_text = "Điểm giữa kỳ 2 (0-20)"


        if feature in categorical_options:
            input_data[feature] = target_col.selectbox(
                f"Chọn {feature}:", 
                options=categorical_options[feature],
                key=feature,
                help=help_text
            )
        elif feature in ['G1', 'G2']:
             input_data[feature] = target_col.number_input(
                f"Nhập điểm {feature}:", 
                min_value=0, max_value=20, value=10, step=1, key=feature, help=help_text
            )
        elif feature == 'age':
            input_data[feature] = target_col.number_input(
                f"Nhập {feature} (tuổi):", 
                min_value=15, max_value=22, value=16, step=1, key=feature, help=help_text
            )
        elif feature == 'absences':
             input_data[feature] = target_col.number_input(
                f"Nhập {feature} (buổi vắng):", 
                min_value=0, max_value=93, value=0, step=1, key=feature, help=help_text
            )
        elif feature in ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']:
            min_val, max_val, default_val = 1, 5, 3 
            if feature == 'failures': max_val, default_val, min_val = 4, 0, 0 # failures có thể từ 0-4
            elif feature == 'traveltime': max_val, default_val = 4, 1
            elif feature == 'studytime': max_val, default_val = 4, 2
            
            input_data[feature] = target_col.number_input(
                f"Nhập {feature}:",
                min_value=min_val,
                max_value=max_val, 
                value=default_val, 
                step=1, key=feature, help=help_text
            )
        else: 
            # Dành cho các feature số khác không có trong danh sách trên (ít khả năng xảy ra)
            input_data[feature] = target_col.number_input(f"Nhập {feature}:", value=0, key=feature, help=help_text)
        
        if caption_text:
            target_col.caption(caption_text)

    submit_button = st.form_submit_button(label="🚀 Dự đoán Nguy cơ", use_container_width=True)

# --- Xử lý và hiển thị kết quả ---
if submit_button:
    # Sắp xếp lại input_data theo thứ tự của feature_columns để đảm bảo tính nhất quán
    ordered_input_data = {col: input_data[col] for col in feature_columns}
    input_df = pd.DataFrame([ordered_input_data])

    st.markdown("---")
    st.subheader("📈 Kết quả Dự đoán:")
    try:
        prediction_label = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0] 

        risk_status_str = "⚠️ Nguy cơ cao (G3 < 10)" if prediction_label == 1 else "✅ An toàn (G3 >= 10)"
        proba_risk = prediction_proba[1] # Xác suất là Nguy cơ cao (lớp 1)
        
        # Sử dụng cột để hiển thị kết quả đẹp hơn
        res_col1, res_col2 = st.columns([2,3])

        with res_col1:
            if prediction_label == 1:
                st.error(f"**Trạng thái: {risk_status_str}**")
            else:
                st.success(f"**Trạng thái: {risk_status_str}**")
        
        with res_col2:
            st.metric(label="Xác suất là 'Nguy cơ cao'", value=f"{proba_risk:.2%}")
            st.progress(proba_risk)
        
        st.markdown("---")
        st.subheader("📝 Lời khuyên:")
        if proba_risk > 0.7:
            st.warning("Học sinh này có nguy cơ **rất cao** đạt kết quả học tập thấp. Cần có sự quan tâm đặc biệt, tìm hiểu nguyên nhân và xây dựng kế hoạch hỗ trợ cụ thể từ gia đình và nhà trường.")
        elif proba_risk > 0.4:
             st.info("Học sinh này có **dấu hiệu nguy cơ**. Nên theo dõi sát sao hơn, khuyến khích tinh thần học tập và tìm hiểu xem em có gặp khó khăn gì không để hỗ trợ kịp thời.")
        else:
            st.balloons()
            st.info("Học sinh này có vẻ đang học tập tốt. Hãy tiếp tục duy trì và khuyến khích thêm!")

    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán: {e}")
        st.error("Vui lòng kiểm tra lại các giá trị đầu vào đã cung cấp.")

# --- Footer ---
st.markdown("---")
st.caption("Hackathon AI - 01/06/2025 | Dự án: Dự đoán Nguy cơ Học tập của Học sinh")