"""
Spam Email & Phishing Classifier Service & Web Application.
Supports both interactive Streamlit Web UI (localhost) and CLI execution.
"""

from datetime import datetime
from pathlib import Path
import re
import sys
import joblib
import pandas as pd
import streamlit as st

# Dynamic Path Resolution
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
LEGACY_MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"

# High-risk trigger keywords for lexical analysis
TRIGGER_KEYWORDS = [
    "lottery", "winner", "prize", "claim", "urgent", "congratulations",
    "selected", "cash", "free", "award", "reward", "verify", "password",
    "account", "suspended", "security", "bitcoin", "crypto", "profit",
    "guaranteed", "click here", "call now", "limited time", "discount",
    "deposit", "exclusive", "jackpot", "millionaire", "risk-free"
]

SAMPLE_PRESETS = {
    "🎰 Lottery Jackpot": (
        "Congratulations! You have been selected as the official winner of our $1,000,000 international sweepstakes. "
        "Text WIN to 77890 or call now to claim your cash reward immediately!"
    ),
    "💼 Team Standup (Ham)": (
        "Hi Rohit, could you please review the attached machine learning pipeline report and let me know your feedback "
        "before tomorrow's sprint review meeting?"
    ),
    "⚠️ Phishing Alert": (
        "URGENT: Your bank account has been flagged for unauthorized activity and is temporarily suspended. "
        "Please verify your credentials immediately at http://secure-portal-auth.com to prevent termination."
    ),
    "📦 Delivery Update (Ham)": (
        "Your package order #94821 has been dispatched with express tracking. "
        "Estimated delivery is today between 2:00 PM and 5:00 PM. No signature required."
    ),
    "💎 Crypto 100x Scheme": (
        "Guaranteed 500% profit in 24 hours with our AI algorithmic crypto trading bot! "
        "Deposit $100 today and watch your funds grow risk-free. Only 10 spots left!"
    ),
}


def load_model():
    """Loads the serialized spam classification pipeline."""
    target_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not target_path.exists():
        return None
    return joblib.load(target_path)


def detect_triggers(text: str) -> list[str]:
    """Finds suspicious trigger keywords contained in the text."""
    lower_text = text.lower()
    found = [kw for kw in TRIGGER_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', lower_text)]
    return found


def predict_message(model, message: str):
    """Predicts label and probability for a given message string."""
    cleaned_message = message.strip()
    if not cleaned_message:
        return None, None, 0.0, 0.0, "Please enter a non-empty message."
    if len(cleaned_message) < 3:
        return None, None, 0.0, 0.0, "Message is too short to classify."

    prediction = model.predict([cleaned_message])[0]
    probabilities = model.predict_proba([cleaned_message])[0]

    ham_prob = probabilities[0] * 100
    spam_prob = probabilities[1] * 100

    confidence = probabilities[prediction] * 100
    label = "SPAM" if prediction == 1 else "NOT SPAM"

    return label, confidence, ham_prob, spam_prob, None


def run_streamlit_app():
    """Renders the Enhanced Streamlit Web Application interface on localhost."""
    try:
        st.set_page_config(
            page_title="Spam & Phishing Shield",
            page_icon="📧",
            layout="wide"
        )
    except Exception:
        pass

    # Header using native Streamlit elements
    st.title("📧 Spam & Phishing Detection Shield")
    st.caption("TF-IDF NLP Vectorizer + Multinomial Naive Bayes • Real-Time Classification Engine")
    st.write(
        "Detect deceptive spam, phishing exploits, and promotional fraud in incoming emails or SMS communications with high-speed natural language inference."
    )

    model = load_model()
    if model is None:
        st.error("⚠️ Trained model pipeline not found. Please train the model by running `python spam-email/src/train.py`.")
        st.stop()

    st.markdown("---")

    # Preset Quick Samples
    st.write("🎯 **Quick Scenario Tests:** *(Click any button to populate sample message)*")
    preset_cols = st.columns(len(SAMPLE_PRESETS))
    for idx, (p_title, p_content) in enumerate(SAMPLE_PRESETS.items()):
        with preset_cols[idx]:
            if st.button(p_title, key=f"spam_preset_{idx}", use_container_width=True):
                st.session_state["msg_input"] = p_content

    # Input Box
    default_text = st.session_state.get("msg_input", "")
    message = st.text_area(
        "📝 Inspect Message / Email Content:",
        value=default_text,
        height=140,
        placeholder="Type or paste incoming email, SMS, or message text here to evaluate...",
        key="active_msg_box"
    )

    char_len = len(message.strip()) if message else 0
    word_len = len(message.strip().split()) if message else 0
    st.caption(f"📏 Character Count: **{char_len}** | Word Count: **{word_len}**")

    btn_col1, btn_col2, _ = st.columns([2, 1, 3])
    classify_clicked = btn_col1.button("🔍 Analyze & Classify Message", type="primary", use_container_width=True)
    clear_clicked = btn_col2.button("🧹 Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["msg_input"] = ""
        st.rerun()

    # Inference Execution
    if classify_clicked or (message and "last_classified_msg" in st.session_state and st.session_state["last_classified_msg"] == message):
        label, confidence, ham_prob, spam_prob, error = predict_message(model, message)
        if error:
            st.warning(error)
        else:
            st.session_state["last_classified_msg"] = message
            triggers = detect_triggers(message)

            # Store in Session History
            if "spam_history" not in st.session_state:
                st.session_state["spam_history"] = []

            history = st.session_state["spam_history"]
            if not history or history[0]["message"] != message[:60]:
                history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "message": message[:60] + ("..." if len(message) > 60 else ""),
                    "label": label,
                    "confidence": f"{confidence:.1f}%",
                    "spam_prob": f"{spam_prob:.1f}%"
                })
                st.session_state["spam_history"] = history[:5]

            st.markdown("---")
            st.subheader("🎯 Classification Result")

            # Native Streamlit Container - 100% immune to HTML escaping bugs
            with st.container(border=True):
                if label == "SPAM":
                    st.error("🚨 **DETECTED: SPAM / PHISHING**\n\nHigh probability of deceptive or malicious content. Avoid clicking unknown links or sharing sensitive info.")
                else:
                    st.success("🛡️ **VERIFIED: LEGITIMATE (HAM)**\n\nClean conversational or transactional message. No prominent malicious patterns detected.")

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Overall Confidence", f"{confidence:.1f}%")
                col_m2.metric("Spam Risk Probability", f"{spam_prob:.1f}%")
                col_m3.metric("Safe Ham Probability", f"{ham_prob:.1f}%")

                st.progress(float(spam_prob / 100.0), text=f"📊 Threat Index: {spam_prob:.1f}% Spam Probability")

            # Trigger Keywords Section
            if triggers:
                with st.container(border=True):
                    st.warning(f"⚠️ **Suspicious Trigger Keywords Detected ({len(triggers)}):** " + ", ".join([f"`{kw}`" for kw in triggers]))
            else:
                st.info("✅ No common high-risk spam keyword patterns identified in vocabulary scan.")

    # Session Recent History Table
    if "spam_history" in st.session_state and st.session_state["spam_history"]:
        with st.expander("🕒 Recent Predictions in this Session", expanded=False):
            st.dataframe(
                pd.DataFrame(st.session_state["spam_history"]),
                use_container_width=True,
                hide_index=True
            )


def run_cli():
    """Runs prediction via Command Line Interface."""
    print("========================================")
    print("   Spam & Phishing Shield CLI Runner    ")
    print("========================================")

    model = load_model()
    if model is None:
        print("Error: Model file not found. Please run python train.py first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        label, confidence, ham_prob, spam_prob, error = predict_message(model, message)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Message   : \"{message}\"")
            print(f"Verdict   : {label}")
            print(f"Confidence: {confidence:.2f}% (Spam: {spam_prob:.1f}%, Ham: {ham_prob:.1f}%)")
        return

    try:
        message = input("Enter an email or message: ")
        label, confidence, ham_prob, spam_prob, error = predict_message(model, message)
        if error:
            print(error)
        else:
            print(f"\nVerdict   : {label}")
            print(f"Confidence: {confidence:.2f}% (Spam: {spam_prob:.1f}%, Ham: {ham_prob:.1f}%)")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _st_runtime_exists
        is_running_in_streamlit = _st_runtime_exists()
    except (ImportError, ModuleNotFoundError):
        is_running_in_streamlit = False

    if is_running_in_streamlit:
        run_streamlit_app()
    else:
        run_cli()
