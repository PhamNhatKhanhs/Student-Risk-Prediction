# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src import config

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo biến mục tiêu 'risk_status'."""
    df_copy = df.copy()
    df_copy[config.NEW_TARGET_COLUMN] = (df_copy[config.TARGET_COLUMN] < config.RISK_THRESHOLD).astype(int)
    print(f"Biến mục tiêu '{config.NEW_TARGET_COLUMN}' đã được tạo.")
    # print(df_copy[config.NEW_TARGET_COLUMN].value_counts(normalize=True))
    return df_copy

def get_feature_and_target(df: pd.DataFrame):
    """Tách features (X) và target (y)."""
    # Loại bỏ cột điểm gốc G3 và cột mục tiêu mới tạo ra khỏi features
    # Các cột G1, G2 có thể giữ lại hoặc loại bỏ tùy chiến lược
    # Ở đây ta giữ G1, G2
    X = df.drop([config.TARGET_COLUMN, config.NEW_TARGET_COLUMN], axis=1)
    y = df[config.NEW_TARGET_COLUMN]
    print("Features (X) và Target (y) đã được tách.")
    return X, y

def get_feature_types(X: pd.DataFrame):
    """Xác định các cột số và cột phân loại."""
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    return numerical_features, categorical_features

def create_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """Tạo ColumnTransformer để tiền xử lý features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # sparse_output=False để dễ xem
        ],
        remainder='passthrough'
    )
    print("Preprocessor đã được tạo.")
    return preprocessor

if __name__ == '__main__':
    # Test thử các hàm
    from src.data_loader import load_dataset
    df_raw = load_dataset()
    if df_raw is not None:
        df_processed = create_target_variable(df_raw)
        X_data, y_data = get_feature_and_target(df_processed)
        num_cols, cat_cols = get_feature_types(X_data)
        preprocessor_obj = create_preprocessor(num_cols, cat_cols)
        
        # Thử fit_transform preprocessor
        X_train_sample, _, _, _ = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        X_train_transformed_sample = preprocessor_obj.fit_transform(X_train_sample)
        print(f"\nKích thước X_train sau khi transform: {X_train_transformed_sample.shape}")
        # print("Một vài dòng X_train sau transform:")
        # print(pd.DataFrame(X_train_transformed_sample, columns=preprocessor_obj.get_feature_names_out()).head())