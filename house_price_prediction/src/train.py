"""
House Price Prediction Model Training Script.
Trains a RandomForestRegressor on housing data and saves the model and feature column names.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Define categorical features and their fixed order of categories
CATEGORICAL_CATEGORIES = {
    "mainroad": ["no", "yes"],
    "guestroom": ["no", "yes"],
    "basement": ["no", "yes"],
    "hotwaterheating": ["no", "yes"],
    "airconditioning": ["no", "yes"],
    "prefarea": ["no", "yes"],
    "furnishingstatus": ["furnished", "semi-furnished", "unfurnished"],
}


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical variables using fixed categories and creates dummy features."""
    df_copy = df.copy()
    for col, categories in CATEGORICAL_CATEGORIES.items():
        if col in df_copy.columns:
            df_copy[col] = pd.Categorical(df_copy[col], categories=categories)

    encoded_df = pd.get_dummies(df_copy, drop_first=True, dtype=int)
    return encoded_df


def train():
    # Resolve dynamic paths relative to this script
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data.csv"
    models_dir = base_dir / "models"
    model_path = models_dir / "house_price_model.pkl"
    features_path = models_dir / "feature_columns.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load dataset
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()

    # Separate target and features
    X = df.drop("price", axis=1)
    y = df["price"]

    # Preprocess features
    X_encoded = preprocess_features(X)
    feature_columns = X_encoded.columns.tolist()

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("--- Model Evaluation ---")
    print(f"Mean Absolute Error: {mae:,.2f}")
    print(f"Root Mean Squared Error: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Ensure models directory exists and save artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(feature_columns, features_path)

    print("\nModel and feature columns saved successfully:")
    print(f"- Model: {model_path}")
    print(f"- Feature Columns: {features_path}")


if __name__ == "__main__":
    train()
