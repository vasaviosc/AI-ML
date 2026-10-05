# House Price Prediction

Predicts house price from area, rooms, and amenities using a scikit-learn model, served through a Streamlit UI.

## Setup

```bash
cd house_price_prediction
pip install -r requirements.txt
```

## Train (optional — a trained model is already committed in `models/`)

Run `src/train.ipynb` with the notebook's working directory set to `src/` (the default for Jupyter). It reads `../data.csv` and writes `../models/house_price_model.pkl` and `../models/feature_columns.pkl`.

## Run the app

```bash
streamlit run src/predict.py
```

Run this from any directory — `predict.py` resolves `models/` relative to its own file location, not the current working directory.
