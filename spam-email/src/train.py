import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("data/spam.csv")

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

x = data["message"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

#good to use a pipeline to do the vectorization and classification at the same time
#but it is also good to use stop words to remove words which add unnecessary weightage e.g. 'and, but, to' 
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english')),
    ("classifier", LogisticRegression(max_iter=1000, random_state=29))
])

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))

print("Model trained successfully!")
print()
print("Model Evaluation")
print("-----------------")
print("Accuracy :", round(accuracy, 2))
print()

from sklearn.model_selection import GridSearchCV

# Defined a parameter grid to compare both classifiers inside the pipeline
param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
        'classifier__C': [0.1, 1.0, 10.0],
        'vectorizer__ngram_range': [(1, 1), (1, 2)]
    },
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.1, 0.5, 1.0],
        'vectorizer__ngram_range': [(1, 1), (1, 2)]
    }
]

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

joblib.dump(model, "model/spam_model.pkl")

print()
print("Model saved successfully!")