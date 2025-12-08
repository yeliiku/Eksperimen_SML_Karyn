import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_heart_dataset(csv_path: str):
    df = pd.read_csv(csv_path, na_values='?')
    df = df.dropna()

    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop(columns=['target'])
    y = df['target']

    num_features = ['age','trestbps','chol','thalach','oldpeak']
    cat_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_heart_dataset("data/heart.csv")

    if hasattr(X_train, "toarray"):
        np.savetxt("output/processed_dataset.csv", X_train.toarray(), delimiter=",")
    else:
        np.savetxt("output/processed_dataset.csv", X_train, delimiter=",")

    print("Preprocessing selesai. File tersimpan di output/processed_dataset.csv")
