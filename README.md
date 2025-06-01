# Dự án Dự đoán Nguy cơ Học tập của Học sinh

Dự án này nhằm xây dựng một mô hình Machine Learning để dự đoán sớm những học sinh/sinh viên có nguy cơ đạt kết quả học tập thấp dựa trên các yếu tố đầu vào từ bộ dữ liệu "Student Performance" của UCI.

## Cấu trúc thư mục

student_risk_predictor/
├── data/
│   └── student-mat.csv       # Dữ liệu gốc
├── notebooks/
│   └── student_risk_exploration.ipynb # Notebook khám phá dữ liệu (tùy chọn)
├── src/
│   ├── __init__.py
│   ├── config.py             # Cấu hình
│   ├── data_loader.py        # Tải dữ liệu
│   ├── preprocessing.py      # Tiền xử lý
│   ├── model_trainer.py      # Huấn luyện và đánh giá mô hình
│   ├── evaluate.py           # Hàm đánh giá, vẽ biểu đồ
│   └── predict_utils.py      # Hàm dự đoán (tùy chọn)
├── main.py                   # Script chính
├── requirements.txt          # Thư viện cần thiết
└── README.md                 # File này

## Cách cài đặt và chạy

1.  **Clone repository (nếu có):**
    ```bash
    git clone [your-repo-link]
    cd student_risk_predictor
    ```
2.  **Tạo môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Linux/macOS
    # venv\Scripts\activate   # Trên Windows
    ```
3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Đặt file dữ liệu:**
    Tải file `student-mat.csv` (ví dụ từ [đây](https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv) hoặc nguồn UCI gốc) và đặt vào thư mục `data/`.

5.  **Chạy dự án:**
    ```bash
    python main.py
    ```

## Công nghệ sử dụng
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn