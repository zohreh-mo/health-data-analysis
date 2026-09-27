import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
# 2. BASELINE (Before Feature Engineering)
# ============================================
print("\n" + "="*70)
print("📊 BASELINE (Before Feature Engineering)")
print("="*70)

X_base = df.drop(columns=["Diabetes_012", "target"])
y = df["Diabetes_012"]

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_results = []

for name, model in {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    baseline_results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    })
    print(f"   {name}: Acc={accuracy_score(y_test, y_pred):.4f}, F1={f1_score(y_test, y_pred):.4f}, AUC={roc_auc_score(y_test, y_prob):.4f}")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
print("\n" + "="*70)
print("🔧 APPLYING FEATURE ENGINEERING")
print("="*70)

df_fe = df.copy()

# Feature 1: BMI × Age
df_fe["BMI_Age"] = df_fe["BMI"] * df_fe["Age"]
print("✅ Feature 1: BMI_Age = BMI × Age")

# Feature 2: HighBP + HighChol
df_fe["BP_Chol"] = df_fe["HighBP"] + df_fe["HighChol"]
print("✅ Feature 2: BP_Chol = HighBP + HighChol")

# Feature 3: Age Group
df_fe["Age_Group"] = pd.cut(df_fe["Age"], bins=[0, 4, 8, 13], labels=[0, 1, 2]).astype(float).fillna(0).astype(int)
print("✅ Feature 3: Age_Group (0=young, 1=middle, 2=old)")

# Feature 4: BMI Category
df_fe["BMI_Category"] = pd.cut(df_fe["BMI"], bins=[0, 25, 30, 100], labels=[0, 1, 2]).astype(float).fillna(0).astype(int)
print("✅ Feature 4: BMI_Category (0=normal, 1=overweight, 2=obese)")

# Feature 5: Health Score
df_fe["Health_Score"] = df_fe["GenHlth"] + df_fe["PhysHlth"]/30 + df_fe["MentHlth"]/30
print("✅ Feature 5: Health_Score = GenHlth + PhysHlth/30 + MentHlth/30")

# Feature 6: Lifestyle Score
df_fe["Lifestyle_Score"] = df_fe["Smoker"] + df_fe["HvyAlcoholConsump"] + (1 - df_fe["PhysActivity"])
print("✅ Feature 6: Lifestyle_Score = Smoker + Alcohol + (1 - PhysActivity)")

# Feature 7: Comorbidity Count
df_fe["Comorbidity"] = df_fe["HighBP"] + df_fe["HighChol"] + df_fe["Stroke"] + df_fe["HeartDiseaseorAttack"]
print("✅ Feature 7: Comorbidity = HighBP + HighChol + Stroke + HeartDisease")

# ============================================
# 4. AFTER FEATURE ENGINEERING
# ============================================
print("\n" + "="*70)
print("📊 AFTER FEATURE ENGINEERING")
print("="*70)

X_fe = df_fe.drop(columns=["Diabetes_012", "target"])
y_fe = df_fe["Diabetes_012"]

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=42, stratify=y_fe
)

scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled = scaler_fe.transform(X_test_fe)

fe_results = []

for name, model in {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}.items():
    model.fit(X_train_fe_scaled, y_train_fe)
    y_pred = model.predict(X_test_fe_scaled)
    y_prob = model.predict_proba(X_test_fe_scaled)[:, 1]
    
    fe_results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test_fe, y_pred), 4),
        "F1-Score": round(f1_score(y_test_fe, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test_fe, y_prob), 4)
    })
    print(f"   {name}: Acc={accuracy_score(y_test_fe, y_pred):.4f}, F1={f1_score(y_test_fe, y_pred):.4f}, AUC={roc_auc_score(y_test_fe, y_prob):.4f}")

# ============================================
# 5. COMPARISON
# ============================================
print("\n" + "="*70)
print("📊 COMPARISON: Baseline vs Feature Engineering")
print("="*70)

baseline_df = pd.DataFrame(baseline_results)
baseline_df["Version"] = "Baseline"

fe_df = pd.DataFrame(fe_results)
fe_df["Version"] = "Feature Engineering"

comparison = pd.concat([baseline_df, fe_df], ignore_index=True)
comparison = comparison[["Version", "Model", "Accuracy", "F1-Score", "ROC-AUC"]]
print(comparison.to_string(index=False))

comparison.to_csv("feature_engineering_comparison.csv", index=False)
print("\n✅ Saved to feature_engineering_comparison.csv")

# ============================================
# 6. VISUALIZATION
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ["Accuracy", "F1-Score", "ROC-AUC"]

for i, metric in enumerate(metrics):
    pivot = comparison.pivot(index="Model", columns="Version", values=metric)
    pivot.plot(kind="bar", ax=axes[i], color=["steelblue", "coral"])
    axes[i].set_title(f"{metric}: Baseline vs Feature Engineering", fontsize=12)
    axes[i].set_ylabel(metric)
    axes[i].set_ylim(0, 1)
    axes[i].tick_params(axis="x", rotation=15)
    axes[i].legend(title="Version")
    
    for container in axes[i].containers:
        axes[i].bar_label(container, fmt="%.3f", fontsize=9)

plt.tight_layout()
plt.savefig("feature_engineering_comparison.png", dpi=100)
plt.show()

print("\n✅ All done!")
print("📊 Files saved: feature_engineering_comparison.csv, feature_engineering_comparison.png")