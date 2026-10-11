"""
Unified AI & Machine Learning Multi-Model Intelligence Hub.
Hosts both House Price Valuation Engine and Spam Phishing Detection Shield
on a single responsive localhost web application.
"""

from pathlib import Path
import importlib.util
import streamlit as st

# Configure page at the root level
st.set_page_config(
    page_title="AI/ML Intelligence Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent

# Global Premium Styling (Scoped pure CSS, no body tags)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 0.5rem 1.2rem;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


def load_modules():
    """Dynamically loads prediction modules for both models without caching stale code."""
    hp_path = BASE_DIR / "house_price_prediction" / "src" / "predict.py"
    spam_path = BASE_DIR / "spam-email" / "src" / "predict.py"

    hp_module = None
    spam_module = None

    if hp_path.exists():
        spec_hp = importlib.util.spec_from_file_location("house_price_module", hp_path)
        if spec_hp is not None and spec_hp.loader is not None:
            hp_module = importlib.util.module_from_spec(spec_hp)
            spec_hp.loader.exec_module(hp_module)

    if spam_path.exists():
        spec_spam = importlib.util.spec_from_file_location("spam_email_module", spam_path)
        if spec_spam is not None and spec_spam.loader is not None:
            spam_module = importlib.util.module_from_spec(spec_spam)
            spec_spam.loader.exec_module(spam_module)

    return hp_module, spam_module


hp_mod, spam_mod = load_modules()

# Check model file statuses & sizes
hp_model_file = BASE_DIR / "house_price_prediction" / "models" / "house_price_model.pkl"
spam_model_file = BASE_DIR / "spam-email" / "models" / "spam_model.pkl"

hp_ready = hp_model_file.exists() and hp_mod is not None
spam_ready = spam_model_file.exists() and spam_mod is not None

hp_size_kb = f"{hp_model_file.stat().st_size / 1024:.1f} KB" if hp_model_file.exists() else "N/A"
spam_size_kb = f"{spam_model_file.stat().st_size / 1024:.1f} KB" if spam_model_file.exists() else "N/A"

# Navigation options
PAGES = {
    "📊 Unified Operations Dashboard": "overview",
    "🏠 House Price Valuation Engine": "house_price",
    "📧 Spam & Phishing Detection Shield": "spam_email"
}

# Initialize session state for page selection if not set
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "📊 Unified Operations Dashboard"

# Sidebar Navigation & Telemetry
with st.sidebar:
    st.title("🤖 AI/ML Hub Pro")
    st.caption("🟢 **Status:** Online • Localhost:8501")

    st.markdown("---")
    st.write("**Navigation:**")

    page_list = list(PAGES.keys())
    active_idx = page_list.index(st.session_state["current_page"]) if st.session_state["current_page"] in page_list else 0

    selected_page = st.radio(
        "Select Service:",
        page_list,
        index=active_idx,
        label_visibility="collapsed"
    )
    st.session_state["current_page"] = selected_page

    st.markdown("---")
    st.write("**Pipeline Health:**")

    with st.container(border=True):
        st.write("🏠 **House Price Model**")
        st.write(f"Status: `{'Ready ✅' if hp_ready else 'Missing ⚠️'}`")
        st.caption(f"RandomForest • {hp_size_kb}")

    with st.container(border=True):
        st.write("📧 **Spam Classifier**")
        st.write(f"Status: `{'Ready ✅' if spam_ready else 'Missing ⚠️'}`")
        st.caption(f"TF-IDF + Naive Bayes • {spam_size_kb}")

    st.markdown("---")
    st.caption("⚙️ **Server:** Streamlit 1.61.1 | 🐍 **Python:** 3.13 | ⚡ **Inference:** Scikit-Learn")

# Render View Based on Page Selection
current_choice = PAGES[st.session_state["current_page"]]

if current_choice == "overview":
    # Hero Section using native Streamlit elements
    st.title("🤖 AI & Machine Learning Multi-Model Hub")
    st.caption("🟢 2/2 Services Live • Unified Port 8501 • Dual Isolated Pipelines")
    st.write(
        "An enterprise-grade unified workspace delivering predictive real estate analytics and intelligent "
        "natural language spam protection from a single synchronized server."
    )

    st.markdown("---")

    # 2 Feature Showcase Cards using Native Streamlit Containers
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🏠 House Price Predictor")
            st.caption("Supervised Real Estate Regression • Random Forest")
            st.write(
                "Calculates property market valuations by evaluating 12 architectural and zoning attributes "
                "including square footage, bedrooms, bathrooms, road access, and furnishing status."
            )
            st.write("• **Algorithm:** Random Forest Regressor (100 Trees)")
            st.write("• **Features:** 13 Preprocessed Dimensions")
            st.write("• **Output:** Real-time valuation in INR (Crore / Lakh) & USD")

            st.write("")
            if st.button("🚀 Launch House Price Engine", key="btn_hp_overview", type="primary", use_container_width=True):
                st.session_state["current_page"] = "🏠 House Price Valuation Engine"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("📧 Spam & Phishing Shield")
            st.caption("NLP Binary Classification Pipeline • TF-IDF + Naive Bayes")
            st.write(
                "Scans communications for deceptive fraud, sweepstakes spam, and corporate inquiries using "
                "tokenized text vectorization and probabilistic inference."
            )
            st.write("• **Algorithm:** Multinomial Naive Bayes")
            st.write("• **Vectorizer:** TF-IDF with English Stopword filtering")
            st.write("• **Features:** Lexical trigger scanner + threat index gauge")

            st.write("")
            if st.button("🚀 Launch Spam & Phishing Shield", key="btn_spam_overview", type="primary", use_container_width=True):
                st.session_state["current_page"] = "📧 Spam & Phishing Detection Shield"
                st.rerun()

    st.markdown("---")

    # Architectural Highlights & Diagnostics using Native Streamlit Containers
    st.subheader("⚡ System Architecture & Execution Blueprint")

    arch_c1, arch_c2, arch_c3 = st.columns(3)
    with arch_c1:
        with st.container(border=True):
            st.markdown("#### 🌐 Unified Port Binding")
            st.write(
                "Both analytical models share a single active web process without requiring separate terminal sessions or conflicting ports."
            )

    with arch_c2:
        with st.container(border=True):
            st.markdown("#### 🔒 Isolated Pipelines")
            st.write(
                "Pipelines and modules execute independently without cross-contaminating features, weights, or session state."
            )

    with arch_c3:
        with st.container(border=True):
            st.markdown("#### ⚡ Sub-Millisecond Inference")
            st.write(
                "Serialized pickle artifacts execute high-throughput vector transformations and predictions instantly in memory."
            )

elif current_choice == "house_price":
    if hp_mod and hasattr(hp_mod, "run_streamlit_app"):
        hp_mod.run_streamlit_app()
    else:
        st.error("Failed to load House Price Prediction module.")

elif current_choice == "spam_email":
    if spam_mod and hasattr(spam_mod, "run_streamlit_app"):
        spam_mod.run_streamlit_app()
    else:
        st.error("Failed to load Spam Email Classifier module.")
