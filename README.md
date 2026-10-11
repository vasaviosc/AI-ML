# 🤖 AI & Machine Learning Multi-Model Intelligence Hub

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Live%20%26%20Synchronized-success.svg)](https://streamlit.io/)

An enterprise-grade unified machine learning hub serving **House Price Valuation** and **Spam & Phishing Detection** simultaneously on a single localhost web server with zero cross-contamination.

---

## 🌟 Key Features & Capabilities

### 1. 📊 Unified Operations Dashboard
- **Single Port Binding (`localhost:8501`):** Both analytical models share one active server process without port collisions. Streamlit automatically falls back to the next free port (e.g. `8502`) if `8501` is already in use — the exact URL is printed in the terminal.
- **Isolated Model Pipelines:** Each pipeline manages its own serialized weights and preprocessors independently.
- **Live Telemetry & Health Checks:** Real-time monitoring of model sizes, serialized pickle artifacts, and server health.
- **Zero-Restart Navigation:** Switch instantly between services via the sidebar without restarting servers or reloading URLs.

### 2. 🏠 House Price Valuation Engine
- **Architectural & Zoning Attributes:** Evaluates built-up area, bedrooms, bathrooms, stories, parking capacity, furnishing status, A/C, water heating, basement, guest room, main road access, and preferred location zones.
- **⚡ 1-Click Quick Preset Templates:**
  - `🌟 Luxury Villa` (7,500 sq ft, 4 Bed, 3 Bath, Furnished, AC, Prime Sector)
  - `🏢 Urban Flat` (3,200 sq ft, 2 Bed, 2 Bath, Semi-Furnished, Main Road)
  - `🏡 Suburban Home` (4,800 sq ft, 3 Bed, 2 Bath, Basement, Main Road)
  - `🌿 Starter Cottage` (1,800 sq ft, 2 Bed, 1 Bath, Unfurnished)
- **Multi-Category Card Grid:** Clean, structured input layout divided into *Architecture & Scale*, *Interior & Utilities*, and *Location & Connectivity*.
- **Rich Valuation Output:**
  - Real-time valuation in **Indian Rupees** formatted in Lakhs and Crores (e.g., `₹1.25 Crore` / `₹85.40 Lakh`) alongside full numeric amounts.
  - Approximate **USD ($)** conversion.
  - Calculated **Rate Per Square Foot** (`₹ / sq ft`).
- **Dynamic Asset Tier Classification:** Categorizes assets into *💎 Ultra-Luxury Estate*, *🏆 Premium Executive Residence*, *🌟 Standard Suburban Home*, or *🏷️ Economy Starter Property*.
- **Interactive Benchmark Chart:** Compares the active property's specifications directly against dataset baselines.

### 3. 📧 Spam & Phishing Detection Shield
- **High-Speed NLP Pipeline:** Natural language text vectorization using **TF-IDF** paired with a **Multinomial Naive Bayes** classifier (~98% accuracy).
- **🎯 Scenario Quick Tests:**
  - `🎰 Lottery Jackpot` (Deceptive sweepstakes spam)
  - `💼 Team Standup` (Authentic workplace collaboration)
  - `⚠️ Phishing Alert` (Account compromise and fake authentication)
  - `📦 Delivery Update` (Transactional delivery logistics)
  - `💎 Crypto Scheme` (High-yield investment fraud)
- **Real-Time Input Metrics:** Live character and word count tracking.
- **Threat Verdict Card:**
  - High-impact visual alert banners (`🚨 DETECTED: SPAM / PHISHING` vs. `🛡️ VERIFIED: LEGITIMATE (HAM)`).
  - Tri-metric breakdown: **Overall Confidence %**, **Spam Risk Probability %**, and **Safe Ham Probability %**.
  - Visual **Threat Index Progress Bar**.
- **Lexical Trigger Keyword Scanner:** Automatically identifies and highlights high-risk spam words present in the message (e.g., `lottery`, `urgent`, `verify`, `prize`, `cash`, `claim`).
- **Interactive Session History:** Maintains a session log of recent evaluations with timestamps and probability scores.

---

## 🚀 Quick Start Guide

> **All commands below should be run from the repository root** — the outer `AI-ML-main/` folder that contains `app.py` and `run.bat`.

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/AI-ML.git
cd AI-ML-main
pip install -r requirements.txt
```

### 2. Launch the Application
Run the unified hub launcher:

```bash
python -m streamlit run app.py
```

*Or on Windows, simply double-click:*
```powershell
.\run.bat
```

### 3. Access the Portal
Open your web browser and navigate to:
```
http://localhost:8501
```

> **Note:** If port `8501` is already occupied, Streamlit will automatically bind to the next available port (e.g. `8502`). The exact URL is printed in the terminal after launch.

Use the left sidebar radio menu to switch between the **Dashboard**, **House Price Predictor**, and **Spam Classifier**.

---

## 📦 Standalone Execution (Optional)

Each model can also be executed independently if required:

### Run House Price Predictor Standalone:
```bash
python -m streamlit run AI-ML-main/house_price_prediction/src/predict.py
```

### Run Spam Classifier Standalone:
```bash
python -m streamlit run AI-ML-main/spam-email/src/predict.py
```

### Run Spam Classifier via CLI:
```bash
python AI-ML-main/spam-email/src/predict.py "Congratulations! You have won $10,000. Claim now!"
```

---

## 🛠️ Re-training Models

Both models include automated training pipelines that re-fit and serialize fresh weights to `models/`:

### Retrain House Price Random Forest Regressor:
```bash
python AI-ML-main/house_price_prediction/src/train.py
```
*Outputs: `AI-ML-main/house_price_prediction/models/house_price_model.pkl` & `feature_columns.pkl`*

### Retrain Spam Naive Bayes Classifier:
```bash
python AI-ML-main/spam-email/src/train.py
```
*Outputs: `AI-ML-main/spam-email/models/spam_model.pkl`*

---

## 📂 Project Architecture

```
AI-ML-main/                               # Repository Root (run all commands from here)
│
├── app.py                                # Outer Launcher — forwards to inner app.py
├── run.bat                               # 1-Click Windows Launch Script
├── requirements.txt                      # Root-level Dependencies
├── README.md                             # Root Documentation
│
└── AI-ML-main/                           # Core Application Package
    ├── app.py                            # Unified Multi-Model Hub (Streamlit entry point)
    ├── README.md                         # This file
    │
    ├── house_price_prediction/           # Real Estate Valuation Subsystem
    │   ├── data.csv                      # Housing Dataset
    │   ├── requirements.txt              # Subsystem Requirements
    │   ├── models/
    │   │   ├── house_price_model.pkl     # Trained Random Forest Regressor
    │   │   └── feature_columns.pkl       # Feature Alignment Schema
    │   └── src/
    │       ├── train.py                  # Model Training & Evaluation Pipeline
    │       └── predict.py                # Valuation Engine & Interactive UI
    │
    └── spam-email/                       # NLP Classification Subsystem
        ├── data/
        │   └── spam.csv                  # SMS & Email Labeled Dataset
        ├── requirements.txt              # Subsystem Requirements
        ├── models/
        │   └── spam_model.pkl            # Trained TF-IDF + Naive Bayes Pipeline
        └── src/
            ├── train.py                  # NLP Training & Evaluation Pipeline
            └── predict.py                # Detection Shield & CLI Service
```

---

## ⚙️ Tech Stack & Dependencies
- **Core Framework:** [Streamlit](https://streamlit.io/) ≥ 1.30
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) ≥ 1.3 (RandomForestRegressor, MultinomialNB, TfidfVectorizer)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/) ≥ 2.0, [NumPy](https://numpy.org/) ≥ 1.24
- **Visualisation:** [Matplotlib](https://matplotlib.org/) ≥ 3.7, [Seaborn](https://seaborn.pydata.org/) ≥ 0.12
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/) ≥ 1.3
- **Python Runtime:** Python 3.10 – 3.13
