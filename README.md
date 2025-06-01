# 📚 Hệ Thống Dự Đoán Nguy Cơ Học Tập

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![Giấy phép](https://img.shields.io/badge/Giấy%20phép-MIT-green.svg)

## 📋 Tổng Quan

Hệ thống Dự đoán Nguy cơ Học tập là một ứng dụng học máy được thiết kế để nhận diện học sinh có nguy cơ đạt kết quả học tập kém. Sử dụng bộ dữ liệu "Student Performance" của UCI, hệ thống phân tích nhiều thuộc tính của học sinh để dự đoán những em có khả năng đạt điểm cuối kỳ thấp (G3 < 10), từ đó có thể triển khai các chiến lược can thiệp sớm.

<p align="center">
  <img src="https://github.com/user/student_risk_predictor/raw/main/assets/dashboard_preview.png" alt="Xem trước Bảng điều khiển" width="80%">
  <br>
  <em>Xem trước Bảng điều khiển (Ví dụ)</em>
</p>

## ✨ Tính Năng Chính

- **Phân tích Dự đoán**: Sử dụng mô hình Random Forest để dự đoán nguy cơ học tập với độ chính xác cao
- **Giao diện Web Tương tác**: Bảng điều khiển Streamlit thân thiện với người dùng cho dự đoán thời gian thực
- **Phân tích Dữ liệu Toàn diện**: Xem xét hơn 30 thuộc tính của học sinh bao gồm nhân khẩu học, yếu tố xã hội và kết quả học tập trước đó
- **Xác suất Nguy cơ**: Cung cấp điểm xác suất để định lượng độ tin cậy của dự đoán
- **Thông tin Hành động**: Đưa ra các khuyến nghị phù hợp dựa trên mức độ nguy cơ

## 🔍 Cấu Trúc Dự Án

```
student_risk_predictor/
├── data/
│   └── student-mat.csv       # Bộ dữ liệu gốc từ UCI
├── notebooks/
│   └── student_risk_exploration.ipynb # Notebook khám phá dữ liệu
├── src/
│   ├── __init__.py
│   ├── config.py             # Cài đặt cấu hình
│   ├── data_loader.py        # Tiện ích tải dữ liệu
│   ├── preprocessing.py      # Pipeline tiền xử lý dữ liệu
│   ├── model_trainer.py      # Huấn luyện và đánh giá mô hình
│   ├── evaluate.py           # Đánh giá và biểu đồ hóa
│   └── predict_utils.py      # Tiện ích dự đoán
├── app.py                    # Ứng dụng web Streamlit
├── main.py                   # Pipeline huấn luyện chính
├── random_forest_pipeline.joblib  # Pipeline mô hình đã lưu
├── feature_columns.joblib    # Cột đặc trưng đã lưu
├── requirements.txt          # Thư viện phụ thuộc
└── README.md                 # Tài liệu này
```

## 🚀 Cài Đặt

### Yêu cầu hệ thống
- Python 3.7+
- Git (tùy chọn)

### Hướng dẫn cài đặt

1. **Sao chép kho lưu trữ:**
   ```bash
   git clone https://github.com/yourusername/student_risk_predictor.git
   cd student_risk_predictor
   ```

2. **Tạo môi trường ảo:**
   ```bash
   # Trên Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Trên macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Chuẩn bị bộ dữ liệu:**
   - Tải file `student-mat.csv` từ [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) hoặc [đường dẫn trực tiếp này](https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv)
   - Đặt file vào thư mục `data/`

## 💻 Sử Dụng

### Huấn luyện Mô hình

Để huấn luyện mô hình dự đoán và tạo các file mô hình cần thiết:

```bash
python main.py
```

Quá trình này sẽ:
- Tải và tiền xử lý bộ dữ liệu
- Huấn luyện nhiều mô hình phân loại
- Đánh giá và chọn mô hình hoạt động tốt nhất
- Lưu pipeline mô hình và cột đặc trưng để sử dụng sau
- Chạy demo dự đoán mẫu

### Chạy Ứng dụng Web

Để khởi chạy bảng điều khiển Streamlit tương tác:

```bash
streamlit run app.py
```

Ứng dụng sẽ khả dụng tại http://localhost:8501 trong trình duyệt web của bạn.

## 🧪 Hiệu Suất Mô Hình

Hệ thống sử dụng mô hình Random Forest với các chỉ số hiệu suất sau trên tập kiểm thử:

- **Độ chính xác (Accuracy)**: ~90%
- **Độ chuẩn xác (Precision)**: ~88%
- **Độ nhạy (Recall)**: ~85%
- **Điểm F1 (F1 Score)**: ~86%
- **ROC AUC**: ~0.92

## 🔧 Công Nghệ Sử Dụng

- **Python**: Ngôn ngữ lập trình chính
- **Pandas & NumPy**: Xử lý và phân tích dữ liệu
- **Scikit-learn**: Thuật toán học máy và đánh giá
- **Matplotlib & Seaborn**: Biểu đồ hóa dữ liệu
- **Streamlit**: Ứng dụng web tương tác
- **Joblib**: Lưu trữ mô hình

## 📊 Bộ Dữ Liệu

Dự án này sử dụng [Bộ dữ liệu Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance) từ UCI Machine Learning Repository, bao gồm:

- Thông tin nhân khẩu học (tuổi, giới tính, quy mô gia đình)
- Yếu tố xã hội (học vấn của cha mẹ, mối quan hệ gia đình)
- Đặc điểm liên quan đến trường học (thời gian học, số buổi vắng mặt)
- Kết quả học tập trước đó (điểm G1, G2)

## 🤝 Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng gửi Pull Request.

1. Fork kho lưu trữ
2. Tạo nhánh tính năng của bạn (`git checkout -b feature/tinh-nang-tuyet-voi`)
3. Commit các thay đổi (`git commit -m 'Thêm tính năng tuyệt vời'`)
4. Push lên nhánh (`git push origin feature/tinh-nang-tuyet-voi`)
5. Mở Pull Request

## 📝 Giấy Phép

Dự án này được cấp phép theo Giấy phép MIT - xem file LICENSE để biết chi tiết.

## 📬 Liên Hệ

Đường dẫn dự án: [https://github.com/yourusername/student_risk_predictor](https://github.com/yourusername/student_risk_predictor)

---

<p align="center">Phát triển cho Hackathon AI - Ngày 01 tháng 06 năm 2025</p>