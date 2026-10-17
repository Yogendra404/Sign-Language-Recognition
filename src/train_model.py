import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

CSV_PATH = "data/landmarks.csv"
MODEL_PATH = "models/sign_model.pkl"
os.makedirs("models", exist_ok=True)


def normalize(row):
    """63 raw values -> wrist-relative, size-scaled 63 values."""
    pts = np.array(row, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]                      # wrist becomes origin
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale                   # remove hand-size/distance effect
    return pts.flatten()


df = pd.read_csv(CSV_PATH)
print("Samples per sign:\n", df["label"].value_counts(), "\n")

X = np.array([normalize(r) for r in df.drop(columns=["label"]).values])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200,
                               class_weight="balanced",
                               random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4), "\n")
print(classification_report(y_test, pred))

labels = sorted(set(y))
cm = confusion_matrix(y_test, pred, labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)

joblib.dump(model, MODEL_PATH)
print("Model saved to", MODEL_PATH)
print("Confusion matrix saved to models/confusion_matrix.png")