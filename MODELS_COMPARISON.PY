import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# ============================================
# 1. Load dataset
# ============================================
df = pd.read_parquet("test.parquet")

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

# ============================================
# 2. Prepare data
# ============================================
X = df.drop(columns=["Diabetes_012", "target"])
y = df["Diabetes_012"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test")

# ============================================
# 3. Define models
# ============================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# ============================================
# 4. Train and evaluate
# ============================================
results = []

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Cross-validation (5-fold)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    
    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "CV Mean": round(cv_scores.mean(), 4),
        "CV Std": round(cv_scores.std(), 4)
    })
    
    print(f"   ✅ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# ============================================
# 5. Results table
# ============================================
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("📊 MODEL COMPARISON RESULTS")
print("="*70)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv("model_comparison.csv", index=False)
print("\n✅ Results saved to model_comparison.csv")

# ============================================
# 6. Visualization: Model Comparison
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ["Accuracy", "F1-Score", "ROC-AUC"]
colors = ["steelblue", "coral", "mediumseagreen"]

for i, metric in enumerate(metrics):
    axes[i].bar(results_df["Model"], results_df[metric], color=colors[i])
    axes[i].set_title(f"{metric} Comparison", fontsize=14)
    axes[i].set_ylabel(metric)
    axes[i].set_ylim(0, 1)
    axes[i].tick_params(axis="x", rotation=15)
    
    # Add value labels
    for j, v in enumerate(results_df[metric]):
        axes[i].text(j, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=100)
plt.show()

# ============================================
# 7. Best model
# ============================================
best_model = results_df.loc[results_df["Accuracy"].idxmax()]
print("\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']}")
print(f"   F1-Score: {best_model['F1-Score']}")
print(f"   ROC-AUC: {best_model['ROC-AUC']}")
print("="*70)

print("\n✅ All done!")