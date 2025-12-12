import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb

# Load Titanic training dataset from local CSV
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df = pd.concat([df_train, df_test]).reset_index(drop=True)

# Select useful columns for this example
cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[cols]

# Drop rows with missing values (for now, keep it simple)
df = df.dropna()

# Encode 'Sex' as 0 = male, 1 = female
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Encode 'Embarked' as 0 = S, 1 = C, 2 = Q
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Separate features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Convert y to integer (XGBoost expects integer labels)
y = y.astype(int)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))
print()

# Visualize Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw confusion matrix
ConfusionMatrixDisplay(cm, display_labels=["Did Not Survive", "Survived"]).plot(
    ax=axes[0], cmap="Blues", values_format='d'
)
axes[0].set_title("XGBoost - Raw Confusion Matrix")

# Normalized confusion matrix
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
ConfusionMatrixDisplay(cm_normalized, display_labels=["Did Not Survive", "Survived"]).plot(
    ax=axes[1], cmap="Greens", values_format='.2f'
)
axes[1].set_title("XGBoost - Normalized Confusion Matrix")

plt.tight_layout()
plt.savefig("xgboost_confusion_matrix.png", dpi=150, bbox_inches='tight')
print("Confusion matrix plot saved as 'xgboost_confusion_matrix.png'")
plt.show()

