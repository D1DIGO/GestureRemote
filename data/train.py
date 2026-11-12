import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("dataset/gestures.csv")

X = df.drop(columns=["label"])
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=4,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))


# Save model
joblib.dump(model, "models/gesture_classifier.pkl")

print("\nModel saved!")
