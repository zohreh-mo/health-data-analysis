import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
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

# === Figure 1: Age Distribution ===
plt.figure(figsize=(10, 6))
sns.histplot(df["Age"], bins=13, kde=True, color="steelblue")
plt.title("Age Distribution in the Dataset", fontsize=14)
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# === Figure 2: Diabetes Distribution ===
plt.figure(figsize=(8, 6))
df["Diabetes_012"].value_counts().plot(kind="bar", color=["green", "orange", "red"])
plt.title("Diabetes Distribution (0=Healthy, 1=Pre-diabetes, 2=Diabetes)", fontsize=14)
plt.xlabel("Diabetes Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("diabetes_distribution.png")
plt.show()

# === Figure 3: BMI vs Diabetes ===
plt.figure(figsize=(10, 6))
sns.boxplot(x="Diabetes_012", y="BMI", data=df, palette="Set2")
plt.title("Relationship between BMI and Diabetes", fontsize=14)
plt.xlabel("Diabetes Status")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("bmi_diabetes.png")
plt.show()

# === Figure 4: Correlation Matrix ===
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Variables", fontsize=14)
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

print("\n✅ All figures created and saved!")