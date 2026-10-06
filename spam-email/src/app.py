import re
import joblib
import streamlit as st
from pathlib import Path


# ============================================================
# PROJECT AND MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .suspicious-term {
        display: inline-block;
        padding: 7px 12px;
        margin: 4px 4px 4px 0;
        border-radius: 8px;
        background-color: #3a1f1f;
        color: #ff6b6b;
        font-size: 16px;
        font-weight: 700;
    }

    .reason-item {
        font-size: 16px;
        margin: 9px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ANALYZE MESSAGE
# ============================================================

def analyze_message(message):

    message_lower = message.lower()

    categories = []
    suspicious_words = []

    # Suspicious categories and their risk weights
    patterns = {

        "Urgent language": {
            "words": [
                "urgent",
                "immediately",
                "act now",
                "limited time",
                "expires",
                "hurry"
            ],
            "weight": 1
        },

        "Financial language": {
            "words": [
                "money",
                "cash",
                "prize",
                "reward",
                "winner",
                "won",
                "refund",
                "₹",
                "$"
            ],
            "weight": 1
        },

        "Account / verification": {
            "words": [
                "otp",
                "password",
                "pin",
                "verify",
                "verification",
                "account",
                "login"
            ],
            "weight": 2
        },

        "Promotional language": {
            "words": [
                "free",
                "bonus",
                "offer",
                "claim",
                "congratulations",
                "discount"
            ],
            "weight": 1
        }
    }


    # --------------------------------------------------------
    # Check suspicious categories
    # --------------------------------------------------------

    for category, details in patterns.items():

        found_words = []

        for word in details["words"]:

            # Currency symbols
            if word in ["₹", "$"]:

                if word in message_lower:
                    found_words.append(word)

            # Normal words / phrases
            else:

                pattern = r"\b" + re.escape(word) + r"\b"

                if re.search(pattern, message_lower):
                    found_words.append(word)


        if found_words:

            categories.append({
                "name": category,
                "weight": details["weight"]
            })

            suspicious_words.extend(found_words)


    # ========================================================
    # CHECK FOR EXTERNAL LINKS
    # ========================================================

    link_pattern = r"https?://[^\s]+|www\.[^\s]+"

    if re.search(link_pattern, message_lower):

        categories.append({
            "name": "External link",
            "weight": 2
        })

        suspicious_words.append("external link")


    # ========================================================
    # REMOVE DUPLICATE SUSPICIOUS TERMS
    # ========================================================

    suspicious_words = list(
        dict.fromkeys(suspicious_words)
    )


    # ========================================================
    # CALCULATE RISK SCORE
    # ========================================================

    risk_score = sum(
        category["weight"]
        for category in categories
    )


    # Number of detected suspicious categories
    risk_signals = len(categories)


    # ========================================================
    # MESSAGE STATISTICS
    # ========================================================

    characters = len(message)

    words = len(message.split())

    links = len(
        re.findall(
            link_pattern,
            message_lower
        )
    )


    return (
        categories,
        suspicious_words,
        risk_score,
        risk_signals,
        characters,
        words,
        links
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "Model not found. Please train the model first."
    )

    st.info(
        "Run: python src/train.py"
    )

    st.stop()


model = joblib.load(MODEL_PATH)


# ============================================================
# MAIN UI
# ============================================================

st.title("📧 Spam Email Detector")

st.write(
    "Analyze an email or message using machine learning "
    "and transparent risk indicators."
)


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Enter your message:",
    height=180,
    placeholder="Type or paste your email/message here..."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Message",
    use_container_width=True
):

    message = message.strip()


    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not message:

        st.warning(
            "Please enter a message."
        )

    elif len(message) < 3:

        st.warning(
            "Message is too short. "
            "Please enter a longer message."
        )

    else:

        # ====================================================
        # ML PREDICTION
        # ====================================================

        prediction = model.predict(
            [message]
        )[0]

        probability = model.predict_proba(
            [message]
        )[0]


        spam_probability = probability[1] * 100

        ham_probability = probability[0] * 100


        if prediction == 1:

            confidence = spam_probability
            result = "🚨 SPAM"

        else:

            confidence = ham_probability
            result = "✅ NOT SPAM"


        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        (
            categories,
            suspicious_words,
            risk_score,
            risk_signals,
            characters,
            words,
            links
        ) = analyze_message(message)


        # ====================================================
        # RISK LEVEL
        #
        # 0-1 points -> LOW
        # 2-3 points -> MEDIUM
        # 4+ points  -> HIGH
        # ====================================================

        if risk_score >= 4:

            risk_level = "🔴 HIGH RISK"

        elif risk_score >= 2:

            risk_level = "🟠 MEDIUM RISK"

        else:

            risk_level = "🟢 LOW RISK"


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.subheader(result)


        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        st.progress(
            int(confidence)
        )


        # ====================================================
        # SPAM / NOT SPAM PROBABILITIES
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )


        with col2:

            st.metric(
                "Not Spam Probability",
                f"{ham_probability:.2f}%"
            )


        st.divider()


        # ====================================================
        # FOUR INFORMATION BOXES
        # ====================================================

        top_left, top_right = st.columns(2)


        # ====================================================
        # TOP LEFT
        # RISK ANALYSIS + SUSPICIOUS TERMS
        # ====================================================

        with top_left:

            with st.container(border=True):

                st.markdown(
                    '<div class="section-title">'
                    '🛡️ Risk Analysis'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.write(
                    f"**Risk Level:** {risk_level}"
                )


                st.write(
                    f"**Risk Score:** {risk_score}"
                )


                st.write(
                    f"**Risk Signals:** {risk_signals}"
                )


                st.markdown(
                    '<div class="section-title">'
                    '🔍 Suspicious Terms'
                    '</div>',
                    unsafe_allow_html=True
                )


                if suspicious_words:

                    terms_html = ""

                    for word in suspicious_words:

                        terms_html += (
                            '<span class="suspicious-term">'
                            + word +
                            '</span>'
                        )


                    st.markdown(
                        terms_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.success(
                        "No suspicious terms detected."
                    )


        # ====================================================
        # TOP RIGHT
        # WHY THIS MESSAGE MAY BE SUSPICIOUS
        # ====================================================

        with top_right:

            with st.container(border=True):

                st.markdown(
                    '<div class="section-title">'
                    '🔎 Why This Message May Be Suspicious'
                    '</div>',
                    unsafe_allow_html=True
                )


                if categories:

                    for category in categories:

                        st.markdown(
                            f'<div class="reason-item">'
                            f'• {category["name"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                else:

                    st.success(
                        "No common suspicious patterns "
                        "were detected."
                    )


        # ====================================================
        # BOTTOM ROW
        # ====================================================

        bottom_left, bottom_right = st.columns(2)


        # ====================================================
        # BOTTOM LEFT
        # MESSAGE ANALYSIS
        # ====================================================

        with bottom_left:

            with st.container(border=True):

                st.markdown(
                    '<div class="section-title">'
                    '📊 Message Analysis'
                    '</div>',
                    unsafe_allow_html=True
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Characters",
                        characters
                    )


                with col2:

                    st.metric(
                        "Words",
                        words
                    )


                with col3:

                    st.metric(
                        "Links",
                        links
                    )


        # ====================================================
        # BOTTOM RIGHT
        # SAFETY RECOMMENDATION
        # ====================================================

        with bottom_right:

            with st.container(border=True):

                st.markdown(
                    '<div class="section-title">'
                    '🛡️ Safety Recommendation'
                    '</div>',
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # HIGH RISK
                # ------------------------------------------------

                if risk_score >= 4:

                    st.warning(
                        "High risk detected. Do not click "
                        "links or provide passwords, OTPs, "
                        "PINs, or financial information. "
                        "Verify the sender through an official "
                        "channel."
                    )


                # ------------------------------------------------
                # MEDIUM RISK
                # ------------------------------------------------

                elif risk_score >= 2:

                    st.warning(
                        "Medium risk detected. Be cautious "
                        "with unexpected promotional or "
                        "prize-related messages. Avoid "
                        "clicking unknown links or sharing "
                        "personal information."
                    )


                # ------------------------------------------------
                # LOW RISK + SPAM
                # ------------------------------------------------

                elif prediction == 1:

                    st.warning(
                        "The model detected spam-like content. "
                        "Avoid responding to unexpected "
                        "messages and verify the sender "
                        "before taking action."
                    )


                # ------------------------------------------------
                # LOW RISK + NOT SPAM
                # ------------------------------------------------

                else:

                    st.warning(
                        "No major suspicious indicators were "
                        "detected. Still verify unexpected "
                        "requests before responding."
                    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()


st.caption(
    "ML Model: Multinomial Naive Bayes + TF-IDF"
)


st.caption(
    "Risk indicators use transparent rule-based "
    "pattern detection."
)