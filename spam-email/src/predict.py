from pathlib import Path
import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "spam_model.pkl"
)

MIN_MESSAGE_LENGTH = 3


def load_model():
    """Load the saved model and handle common loading errors."""
    try:
        return joblib.load(MODEL_PATH)

    except FileNotFoundError:
        print("Error: Trained model not found.")
        print("Run 'python src/train.py' from the spam-email folder first.")

    except PermissionError:
        print("Error: Permission denied while accessing the model.")

    except (EOFError, ValueError, ImportError, ModuleNotFoundError,
            AttributeError, OSError) as error:
        print("Error: Could not load the saved model.")
        print("The file may be corrupted or incompatible.")
        print("Details:", error)

    return None


def main():
    print("Spam Email Classifier")
    print("---------------------")

    model = load_model()

    if model is None:
        return

    try:
        message = input("Enter an email or message: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled. Exiting safely.")
        return

    if not message:
        print("Please enter a non-empty message.")
        return

    if len(message) < MIN_MESSAGE_LENGTH:
        print(
            f"Message is too short. "
            f"Enter at least {MIN_MESSAGE_LENGTH} characters."
        )
        return

    try:
        prediction = model.predict([message])[0]

        if prediction not in (0, 1):
            print("Error: The model returned an unexpected label.")
            return

        if not hasattr(model, "predict_proba"):
            print("Error: Confidence estimates are not supported.")
            return

        probabilities = model.predict_proba([message])[0]
        confidence = probabilities[prediction] * 100

        if prediction == 1:
            print("Prediction: SPAM")
        else:
            print("Prediction: NOT SPAM")

        print("Confidence:", round(confidence, 2), "%")

    except (ValueError, TypeError, AttributeError,
            IndexError, OSError) as error:
        print("Error: Could not classify this message.")
        print("Please check the model and try again.")
        print("Details:", error)


if __name__ == "__main__":
    main()