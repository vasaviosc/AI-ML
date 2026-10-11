"""
House Price Prediction Engine & Interactive Web Application.
Supports both standalone Streamlit execution and Unified Hub embedding.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Resolve dynamic paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "house_price_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.pkl"

# Categorical column definitions matching training pipeline
CATEGORICAL_CATEGORIES = {
    "mainroad": ["no", "yes"],
    "guestroom": ["no", "yes"],
    "basement": ["no", "yes"],
    "hotwaterheating": ["no", "yes"],
    "airconditioning": ["no", "yes"],
    "prefarea": ["no", "yes"],
    "furnishingstatus": ["furnished", "semi-furnished", "unfurnished"],
}

# Pre-defined test property profiles
PRESET_PROFILES = {
    "🌟 Luxury Villa": {
        "area": 7500,
        "bedrooms": 4,
        "bathrooms": 3,
        "stories": 3,
        "parking": 2,
        "mainroad": "yes",
        "guestroom": "yes",
        "basement": "yes",
        "hotwaterheating": "yes",
        "airconditioning": "yes",
        "prefarea": "yes",
        "furnishingstatus": "furnished",
    },
    "🏢 Urban Flat": {
        "area": 3200,
        "bedrooms": 2,
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "prefarea": "yes",
        "furnishingstatus": "semi-furnished",
    },
    "🏡 Suburban Home": {
        "area": 4800,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 2,
        "mainroad": "yes",
        "guestroom": "yes",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "prefarea": "no",
        "furnishingstatus": "semi-furnished",
    },
    "🌿 Starter Cottage": {
        "area": 1800,
        "bedrooms": 2,
        "bathrooms": 1,
        "stories": 1,
        "parking": 0,
        "mainroad": "no",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "no",
        "prefarea": "no",
        "furnishingstatus": "unfurnished",
    },
}


def load_model():
    """Load model and feature columns."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def format_inr(number: float) -> str:
    """Formats a number into Indian Lakhs / Crores notation."""
    if number >= 10000000:
        crores = number / 10000000
        return f"₹{crores:.2f} Crore"
    elif number >= 100000:
        lakhs = number / 100000
        return f"₹{lakhs:.2f} Lakh"
    else:
        return f"₹{number:,.0f}"


def get_luxury_tier(price: float) -> tuple[str, str]:
    """Returns tier title and description based on price valuation."""
    if price >= 8000000:
        return "💎 Ultra-Luxury Estate", "Top-tier prime asset with high-end luxury amenities and premium area zoning."
    elif price >= 5500000:
        return "🏆 Premium Executive Residence", "Spacious upscale property with above-average specifications."
    elif price >= 3500000:
        return "🌟 Standard Suburban Home", "Balanced residential profile ideal for growing modern families."
    else:
        return "🏷️ Economy Starter Property", "Affordable entry-level property with practical core utilities."


def run_streamlit_app():
    """Renders the House Price Prediction UI using 100% native Streamlit components."""
    try:
        st.set_page_config(
            page_title="House Price Valuation Engine",
            page_icon="🏠",
            layout="wide"
        )
    except Exception:
        pass

    # Header using native Streamlit elements
    st.title("🏠 House Price Valuation Engine")
    st.caption("RandomForest ML Regressor • Real-Time Property Valuation & Attribute Analysis")
    st.write(
        "Estimate market valuations in real-time based on architectural footprint, room configuration, interior amenities, and regional zoning factors."
    )

    model, feature_columns = load_model()
    if model is None or feature_columns is None:
        st.error("⚠️ Model artifacts missing. Please train the model by executing `python house_price_prediction/src/train.py`.")
        st.stop()

    st.markdown("---")

    # Preset Profiles Quick Selector
    st.write("⚡ **Quick Preset Templates:** *(Click any preset to auto-fill inputs)*")
    preset_cols = st.columns(4)
    for idx, (p_name, p_vals) in enumerate(PRESET_PROFILES.items()):
        with preset_cols[idx]:
            if st.button(p_name, key=f"hp_preset_{idx}", use_container_width=True):
                for k, v in p_vals.items():
                    st.session_state[f"hp_{k}"] = v

    def get_val(key, default):
        return st.session_state.get(f"hp_{key}", default)

    st.write("")

    # 3-Column Native Streamlit Input Containers
    card1, card2, card3 = st.columns(3)

    with card1:
        with st.container(border=True):
            st.markdown("#### 📐 Architecture & Scale")
            st.caption("Total floor area & structural configuration")

            area = st.number_input(
                "Area (Square Feet)",
                min_value=500,
                max_value=25000,
                value=int(get_val("area", 5000)),
                step=250,
                help="Total living and built-up footprint in sq ft."
            )
            bedrooms = st.slider(
                "Bedrooms",
                min_value=1,
                max_value=6,
                value=int(get_val("bedrooms", 3)),
                step=1
            )
            bathrooms = st.slider(
                "Bathrooms",
                min_value=1,
                max_value=5,
                value=int(get_val("bathrooms", 2)),
                step=1
            )
            stories = st.slider(
                "Stories / Floors",
                min_value=1,
                max_value=4,
                value=int(get_val("stories", 2)),
                step=1
            )
            parking = st.slider(
                "Parking Capacity",
                min_value=0,
                max_value=4,
                value=int(get_val("parking", 2)),
                step=1
            )

    with card2:
        with st.container(border=True):
            st.markdown("#### 🛋️ Interior & Utilities")
            st.caption("Furnishing status, climate & interior features")

            furnish_opts = ["furnished", "semi-furnished", "unfurnished"]
            furnish_default = get_val("furnishingstatus", "furnished")
            furnish_idx = furnish_opts.index(furnish_default) if furnish_default in furnish_opts else 0
            furnishingstatus = st.selectbox(
                "Furnishing Status",
                furnish_opts,
                index=furnish_idx,
                format_func=lambda x: f"{x.capitalize()} Interior"
            )

            ac_default = 0 if get_val("airconditioning", "yes") == "yes" else 1
            airconditioning = st.selectbox(
                "Air Conditioning",
                ["yes", "no"],
                index=ac_default,
                format_func=lambda x: "Yes - Central / Split AC" if x == "yes" else "No AC Installed"
            )

            hw_default = 0 if get_val("hotwaterheating", "no") == "yes" else 1
            hotwaterheating = st.selectbox(
                "Hot Water Heating",
                ["yes", "no"],
                index=hw_default,
                format_func=lambda x: "Yes - Water Heating" if x == "yes" else "No Water Heating"
            )

            gr_default = 0 if get_val("guestroom", "no") == "yes" else 1
            guestroom = st.selectbox(
                "Dedicated Guest Room",
                ["yes", "no"],
                index=gr_default,
                format_func=lambda x: "Yes - Dedicated Suite" if x == "yes" else "No Guest Room"
            )

            base_default = 0 if get_val("basement", "no") == "yes" else 1
            basement = st.selectbox(
                "Basement Storage / Den",
                ["yes", "no"],
                index=base_default,
                format_func=lambda x: "Yes - Basement Available" if x == "yes" else "No Basement"
            )

    with card3:
        with st.container(border=True):
            st.markdown("#### 📍 Location & Connectivity")
            st.caption("Road frontage & sector prestige")

            mr_default = 0 if get_val("mainroad", "yes") == "yes" else 1
            mainroad = st.selectbox(
                "Main Road Frontage",
                ["yes", "no"],
                index=mr_default,
                format_func=lambda x: "Direct Main Road Access" if x == "yes" else "Interior Lane / No Main Road"
            )

            pref_default = 0 if get_val("prefarea", "no") == "yes" else 1
            prefarea = st.selectbox(
                "Prime / Preferred Sector",
                ["yes", "no"],
                index=pref_default,
                format_func=lambda x: "Yes - Prime Sector Zone" if x == "yes" else "Standard Residential Zone"
            )

            st.write("")
            st.info("""
            **🧠 Pipeline Architecture:**
            • Model: Random Forest Regressor
            • Trees: 100 Estimators
            • Encoded Features: 13 Dimensions
            """)

    # Encode input into prediction dataframe
    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    for col, categories in CATEGORICAL_CATEGORIES.items():
        if col in input_data.columns:
            input_data[col] = pd.Categorical(input_data[col], categories=categories)

    input_data_encoded = pd.get_dummies(input_data, drop_first=True, dtype=int)
    input_data_encoded = input_data_encoded.reindex(columns=feature_columns, fill_value=0)

    # Calculate prediction
    prediction = model.predict(input_data_encoded)[0]
    usd_val = prediction / 85.0
    price_per_sqft = prediction / max(area, 1)
    tier_title, tier_desc = get_luxury_tier(prediction)

    st.markdown("---")
    st.subheader("🏡 Valuation Estimate")

    # Native Streamlit Container - 100% immune to HTML escaping bugs
    with st.container(border=True):
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric(
            label="Estimated Price (INR)",
            value=format_inr(prediction),
            delta=f"Exact: ₹{prediction:,.0f}"
        )
        col_v2.metric(
            label="USD Valuation ($)",
            value=f"${usd_val:,.0f} USD"
        )
        col_v3.metric(
            label="Rate Per Sq Ft",
            value=f"₹{price_per_sqft:,.0f} / sq ft",
            delta=f"${(usd_val / max(area, 1)):.1f} / sq ft"
        )

        st.markdown("---")
        st.write(f"**Asset Tier:** `{tier_title}` — {tier_desc}")
        st.info(
            f"📐 **Property Summary:** {area:,} sq ft • 🛏️ {bedrooms} Bed • 🚿 {bathrooms} Bath • "
            f"🏢 {stories} Stories • 🚗 {parking} Parking Spaces • ❄️ AC: {airconditioning.capitalize()} • "
            f"🛣️ Main Road: {mainroad.capitalize()}"
        )

    # Visual Feature Breakdown Section
    with st.expander("📊 Property Comparison Benchmark Chart", expanded=True):
        st.write("Compare the current configuration against average housing baselines:")
        comp_df = pd.DataFrame({
            "Specification": ["Bedrooms", "Bathrooms", "Stories", "Parking", "Area (x1000 sq ft)"],
            "Current Property": [bedrooms, bathrooms, stories, parking, area / 1000.0],
            "Dataset Average": [3.0, 1.3, 1.8, 0.7, 5.15]
        }).set_index("Specification")

        st.bar_chart(comp_df, color=["#38bdf8", "#64748b"])


if __name__ == "__main__":
    run_streamlit_app()