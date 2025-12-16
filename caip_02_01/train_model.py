import pandas as pd
import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # SageMaker passes input as /opt/ml/input/data/train/
    input_path = "/opt/ml/input/data/train/cleaned_titanic.csv"
    df = pd.read_csv(input_path)

    # Split features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)

    print("\n=== Classification Report ===\n")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===\n")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Feature Importances ===\n")
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))

    # Save model
    os.makedirs("/opt/ml/model", exist_ok=True)
    joblib.dump(clf, "/opt/ml/model/model.joblib")

if __name__ == "__main__":
    main()
