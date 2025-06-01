# src/evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd

def get_classification_metrics(y_true, y_pred, model_name="Model"):
    """Tính toán và in các chỉ số phân loại cơ bản."""
    accuracy = accuracy_score(y_true, y_pred)
    # Thêm zero_division=0 để xử lý trường hợp không có TP+FP hoặc TP+FN (tránh warning/error)
    precision = precision_score(y_true, y_pred, zero_division=0) 
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n--- Kết quả đánh giá cho: {model_name} ---")
    print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
    print(f"Precision (cho lớp 1 - Nguy cơ cao): {precision:.4f}")
    print(f"Recall (cho lớp 1 - Nguy cơ cao): {recall:.4f}")
    print(f"F1-score (cho lớp 1 - Nguy cơ cao): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=['An toàn (0)', 'Nguy cơ (1)']))
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name="Model"):
    """Vẽ confusion matrix dưới dạng heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dự đoán An toàn (0)', 'Dự đoán Nguy cơ (1)'], 
                yticklabels=['Thực tế An toàn (0)', 'Thực tế Nguy cơ (1)'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

def plot_feature_importances_custom(pipeline, model_name="Model"):
    """Vẽ feature importances nếu model hỗ trợ."""
    try:
        # Lấy preprocessor và model từ pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier'] # Hoặc 'regressor' 
        
        if hasattr(model, 'feature_importances_'):
            # Lấy tên các features sau khi OneHotEncoding
            feature_names = preprocessor.get_feature_names_out()
            
            importances = model.feature_importances_
            forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

            plt.figure(figsize=(12, max(6, len(forest_importances.head(20)) * 0.3) )) # Điều chỉnh chiều cao biểu đồ
            forest_importances.head(20).plot(kind='barh')
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.gca().invert_yaxis()
            plt.tight_layout() # Tự động điều chỉnh layout cho vừa vặn
            plt.show()
        else:
            print(f"Mô hình {model_name} không hỗ trợ 'feature_importances_'.")
    except Exception as e:
        print(f"Lỗi khi vẽ feature importances cho {model_name}: {e}")