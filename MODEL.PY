import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ============================================
# 1. Load dataset
# ============================================
df = pd.read_parquet("test.parquet")

# Real column names (based on CDC dataset)
column_names = [
    "Age", "BMI", "HighBP", "HighChol", "CholCheck", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Education", "Income", "Diabetes_012", "target"
]

if len(df.columns) == len(column_names):
    df.columns = column_names
    print("✅ Column names updated!")
else:
    print(f"⚠️ Number of columns: {len(df.columns)}")

# ============================================
# 2. Prepare data
# ============================================
# Target: Diabetes_012 (0=Healthy, 1=Pre-diabetes)
X = df.drop(columns=["Diabetes_012", "target"])
y = df["Diabetes_012"]

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test")

# ============================================
# 3. Train Logistic Regression
# ============================================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ============================================
# 4. Evaluate
# ============================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Pre-diabetes"]))

# ============================================
# 5. Confusion Matrix
# ============================================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Pre-diabetes"],
            yticklabels=["Healthy", "Pre-diabetes"])
plt.title("Confusion Matrix - Logistic Regression", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ============================================
# 6. ROC Curve
# ============================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression", fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# ============================================
# 7. Feature Importance
# ============================================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x="Coefficient", y="Feature", data=coefficients, palette="coolwarm")
plt.title("Feature Importance - Logistic Regression", fontsize=14)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\n✅ Model training complete!")
print("📊 Files saved: confusion_matrix.png, roc_curve.png, feature_importance.png")