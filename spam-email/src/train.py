import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#data = pd.read_csv("data/spam.csv")
root = Path(__file__).resolve().parents[1]
data_path = root/"data"/"spam.csv"
model_dir = root/"model"
model_dir.mkdir(exist_ok=True)

data = pd.read_csv(data_path)
data = data.dropna(subset=["label"])
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

x = data["message"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("Model trained successfully!")
print()
print("Model Evaluation")
print("-----------------")
print("Accuracy :", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall   :", round(recall, 2))
print("F1 Score :", round(f1, 2))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, model_dir / "spam_model.pkl")

print()
print("Model saved successfully!")