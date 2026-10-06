from pathlib import Path
import sys
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"

if not MODEL_PATH.exists():
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    print("Please run 'src/train.py' first to train and save the model.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

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
