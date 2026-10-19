import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "spam_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

st.title("📧 Spam Email Classifier")
st.write("Enter an email or message to check whether it is spam or legitimate.")

message = st.text_area(
    "Enter your email or message",
    placeholder="Paste your email or message here...",
    height=180,
    max_chars=10000
)

if st.button("Analyze Message", use_container_width=True):
    message = message.strip()

    if not message:
        st.warning("Please enter a message.")
    elif len(message) < 3:
        st.warning("Message is too short. Enter at least 3 characters.")
    else:
        try:
            prediction = model.predict([message])[0]

            if prediction == 1:
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: NOT SPAM")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([message])[0]
                classes = list(model.classes_)
                probability_by_class = dict(zip(classes, probabilities))

                spam_probability = float(probability_by_class.get(1, 0))
                not_spam_probability = float(probability_by_class.get(0, 0))
                confidence = float(
                    probability_by_class.get(prediction, 0)
                )

                st.metric("Prediction Confidence", f"{confidence * 100:.2f}%")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Spam Probability", f"{spam_probability * 100:.2f}%")
                    st.progress(spam_probability)

                with col2:
                    st.metric("Not Spam Probability", f"{not_spam_probability * 100:.2f}%")
                    st.progress(not_spam_probability)
            else:
                st.info("This model does not support probability estimates.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with st.expander("About the Model"):
    st.write("This dashboard uses the existing trained spam email classifier.")
    st.write("Input: Email or message text")
    st.write("Output: Spam or Not Spam")
    st.write("Feature extraction depends on the preprocessing included in the saved model.")
    st.caption("Predictions and probabilities are estimates, not guarantees.")

st.divider()
st.caption("AIML Open Source Contribution | Streamlit Spam Classifier")
