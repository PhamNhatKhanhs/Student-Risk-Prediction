# src/model_trainer.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src import config, evaluate, preprocessing as preproc # Đổi tên alias để tránh trùng

def train_single_model(X_train, y_train, model_instance, numerical_cols, categorical_cols):
    """Huấn luyện một mô hình đơn lẻ với preprocessor."""
    preprocessor = preproc.create_preprocessor(numerical_cols, categorical_cols) # Gọi hàm từ preproc
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_instance) 
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def train_and_evaluate_models(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols):
    """Huấn luyện và đánh giá nhiều mô hình."""
    models_to_try = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=config.RANDOM_SEED, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=config.RANDOM_SEED, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=config.RANDOM_SEED, class_weight='balanced', n_estimators=100)
    }
    
    trained_pipelines = {}
    all_results_summary = []

    print("\n===== BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH =====")
    for name, model in models_to_try.items():
        print(f"\n--- Đang xử lý mô hình: {name} ---")
        pipeline = train_single_model(X_train, y_train, model, numerical_cols, categorical_cols)
        trained_pipelines[name] = pipeline
        
        y_pred_test = pipeline.predict(X_test)
        
        metrics = evaluate.get_classification_metrics(y_test, y_pred_test, model_name=name)
        evaluate.plot_confusion_matrix_heatmap(y_test, y_pred_test, model_name=name)
        
        # Lưu kết quả để so sánh
        metrics['model_name'] = name
        all_results_summary.append(metrics)
        
        # Vẽ feature importance nếu có
        if name in ["Decision Tree", "Random Forest"]:
             evaluate.plot_feature_importances_custom(pipeline, model_name=name)

    results_df = pd.DataFrame(all_results_summary).sort_values(by='f1', ascending=False)
    print("\n===== TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST =====")
    print(results_df[['model_name', 'accuracy', 'precision', 'recall', 'f1']])
    
    # Chọn mô hình tốt nhất (ví dụ dựa trên F1-score)
    if not results_df.empty:
        best_model_name_from_df = results_df.iloc[0]['model_name']
        print(f"\n>>> Mô hình tốt nhất dựa trên F1-score: {best_model_name_from_df}")
        best_pipeline = trained_pipelines[best_model_name_from_df]
    else:
        print("Không có kết quả mô hình nào để chọn.")
        best_pipeline = None
        best_model_name_from_df = None
        
    return trained_pipelines, best_pipeline, best_model_name_from_df