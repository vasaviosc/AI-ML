"""
Spam Email Classifier Training Script.
Trains a MultinomialNB model with TfidfVectorizer pipeline and saves the serialized model.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train():
    # Resolve dynamic paths relative to this script
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "spam.csv"
    models_dir = base_dir / "models"
    model_path = models_dir / "spam_model.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load dataset and clean missing entries
    data = pd.read_csv(data_path)
    data = data.dropna(subset=["label", "message"])

    # Map labels: ham -> 0, spam -> 1
    data["label"] = data["label"].str.strip().str.lower().map({
        "ham": 0,
        "spam": 1
    })

    X = data["message"]
    y = data["label"]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build Pipeline combining TF-IDF Vectorizer and Multinomial Naive Bayes classifier
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division="warn")
    recall = recall_score(y_test, predictions, zero_division="warn")
    f1 = f1_score(y_test, predictions, zero_division="warn")

    print("=== Model Training Complete ===")
    print("Model Evaluation:")
    print("-----------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Ensure models directory exists and save pipeline
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to: {model_path}")


if __name__ == "__main__":
    train()