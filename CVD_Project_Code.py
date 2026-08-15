
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("CVD_cleaned.csv")

# Target Variable
target = "General_Health"

X = df.drop(columns=[target])
y = df[target]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Models
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=5000)
}

accuracies = {}

for name, model in models.items():
    print("\\n" + "="*60)
    print(f"{name}")
    print("="*60)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    print("Accuracy Score:", acc)
    print("\\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("\\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

# Accuracy Comparison Bar Chart
plt.figure(figsize=(8,5))
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# Hyperparameter Tuning (Random Forest Example)
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_dist = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, 30, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

random_search = RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\\nBest Parameters:")
print(random_search.best_params_)

print("Best CV Accuracy:", random_search.best_score_)

# AI-style Single Prediction
sample = X.iloc[[0]]

prediction = random_search.best_estimator_.predict(sample)

print("\\nSample Input:")
print(sample)

print("\\nPredicted Health Status:")
print(le.inverse_transform(prediction)[0])
