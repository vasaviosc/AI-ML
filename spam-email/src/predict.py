import joblib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
model_dir = root / "model"

try:
    model = joblib.load(model_dir / "spam_model.pkl")
except FileNotFoundError:
    print(f"Model not found")
    raise SystemExit(1)

print("Spam Email Classifier")
print("---------------------")

message = input("Enter an email or message: ").strip()

if not message:
    print("Please enter a message.")
elif len(message) < 3:
    print("Message is too short.")
else:
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0]

    if prediction == 1:
        print("Prediction: SPAM")
        print("Confidence:", round(probability[1] * 100, 2), "%")
    else:
        print("Prediction: NOT SPAM")
        print("Confidence:", round(probability[0] * 100, 2), "%")
