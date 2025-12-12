import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

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

# Convert y to integer
y = y.astype(int)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("HYPERPARAMETER TUNING AND MODEL OPTIMIZATION")
print("=" * 60)
print()

# ============================================================================
# 1. XGBoost with Hyperparameter Tuning
# ============================================================================
print("1. Tuning XGBoost Hyperparameters...")
print("-" * 60)

xgb_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

xgb_random = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_grid,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

xgb_random.fit(X_train, y_train)

print(f"Best XGBoost Parameters: {xgb_random.best_params_}")
print(f"Best XGBoost CV Score: {xgb_random.best_score_:.4f}")
print()

# ============================================================================
# 2. Random Forest with Hyperparameter Tuning
# ============================================================================
print("2. Tuning Random Forest Hyperparameters...")
print("-" * 60)

rf_param_grid = {
    'n_estimators': [100, 200, 300, 500, 700],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf_base = RandomForestClassifier(random_state=42)

rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=rf_param_grid,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_random.fit(X_train, y_train)

print(f"Best Random Forest Parameters: {rf_random.best_params_}")
print(f"Best Random Forest CV Score: {rf_random.best_score_:.4f}")
print()

# ============================================================================
# 3. Evaluate Best Models on Test Set
# ============================================================================
print("=" * 60)
print("FINAL MODEL EVALUATION ON TEST SET")
print("=" * 60)
print()

# Get best models
best_xgb = xgb_random.best_estimator_
best_rf = rf_random.best_estimator_

# Make predictions
y_pred_xgb = best_xgb.predict(X_test)
y_pred_rf = best_rf.predict(X_test)

# Calculate accuracies
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"XGBoost Test Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print()

# Determine best model
if xgb_accuracy >= rf_accuracy:
    best_model = best_xgb
    best_pred = y_pred_xgb
    best_name = "XGBoost"
    best_accuracy = xgb_accuracy
else:
    best_model = best_rf
    best_pred = y_pred_rf
    best_name = "Random Forest"
    best_accuracy = rf_accuracy

print(f"🏆 Best Model: {best_name} with {best_accuracy*100:.2f}% accuracy")
print()

# ============================================================================
# 4. Detailed Results for Best Model
# ============================================================================
print("=" * 60)
print(f"DETAILED RESULTS FOR BEST MODEL ({best_name})")
print("=" * 60)
print()

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
print("Confusion Matrix:")
print(cm)
print()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, best_pred))
print()

# Cross-validation scores for best model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores (5-fold): {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print()

# ============================================================================
# 5. Visualize Confusion Matrix
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw confusion matrix
ConfusionMatrixDisplay(cm, display_labels=["Did Not Survive", "Survived"]).plot(
    ax=axes[0], cmap="Blues", values_format='d'
)
axes[0].set_title(f"{best_name} - Raw Confusion Matrix\nAccuracy: {best_accuracy*100:.2f}%")

# Normalized confusion matrix
cm_normalized = confusion_matrix(y_test, best_pred, normalize='true')
ConfusionMatrixDisplay(cm_normalized, display_labels=["Did Not Survive", "Survived"]).plot(
    ax=axes[1], cmap="Greens", values_format='.2f'
)
axes[1].set_title(f"{best_name} - Normalized Confusion Matrix")

plt.tight_layout()
plt.savefig("optimized_model_confusion_matrix.png", dpi=150, bbox_inches='tight')
print("Confusion matrix plot saved as 'optimized_model_confusion_matrix.png'")
plt.show()

# ============================================================================
# 6. Feature Importance (if XGBoost is best)
# ============================================================================
if best_name == "XGBoost":
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)
    print()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance.png'")
    plt.show()

print("=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)

