# Spam Email Classifier

Classifies a message as spam or ham using a TF-IDF + Naive Bayes pipeline.

## Setup

```bash
cd spam-email
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Run this from the `spam-email` directory — it reads `data/spam.csv` and writes the trained model to `model/spam_model.pkl`, creating the `model/` directory if it doesn't exist yet.

## Predict

```bash
python src/predict.py
```

Also run from the `spam-email` directory, after training (no model is committed to the repo).
