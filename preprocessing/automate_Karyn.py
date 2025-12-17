import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, na_values='?')

    df = df.dropna().reset_index(drop=True)

    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop(columns=['target'])
    y = df['target']

    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    feature_names = (
        num_features +
        list(
            preprocessor
            .named_transformers_['cat']
            .get_feature_names_out(cat_features)
        )
    )

    X_processed_df = pd.DataFrame(
        X_processed.toarray(),
        columns=feature_names
    )

    X_processed_df['target'] = y.values

    return X_processed_df

if __name__ == "__main__":
    processed_df = preprocess_dataset("dataset_raw.csv")
    processed_df.to_csv("processed_dataset.csv", index=False)
