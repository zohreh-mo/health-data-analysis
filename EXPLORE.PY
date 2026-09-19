import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# خوندن دیتاست
df = pd.read_parquet("test.parquet")

# اسم ستون‌های واقعی
column_names = [
    "Age", "BMI", "HighBP", "HighChol", "CholCheck", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Education", "Income", "Diabetes_012", "target"
]

if len(df.columns) == len(column_names):
    df.columns = column_names
    print("✅ اسم ستون‌ها عوض شد!")
else:
    print(f"⚠️ تعداد ستون‌ها: {len(df.columns)}")

# === نمودار ۱: توزیع سن ===
plt.figure(figsize=(10, 6))
sns.histplot(df["Age"], bins=13, kde=True, color="steelblue")
plt.title("توزیع سن در دیتاست", fontsize=14)
plt.xlabel("گروه سنی")
plt.ylabel("تعداد")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# === نمودار ۲: توزیع دیابت ===
plt.figure(figsize=(8, 6))
df["Diabetes_012"].value_counts().plot(kind="bar", color=["green", "orange", "red"])
plt.title("توزیع دیابت (0=سالم، 1=پیش‌دیابت، 2=دیابت)", fontsize=14)
plt.xlabel("وضعیت دیابت")
plt.ylabel("تعداد")
plt.tight_layout()
plt.savefig("diabetes_distribution.png")
plt.show()

# === نمودار ۳: رابطه BMI و دیابت ===
plt.figure(figsize=(10, 6))
sns.boxplot(x="Diabetes_012", y="BMI", data=df, palette="Set2")
plt.title("رابطه BMI و دیابت", fontsize=14)
plt.xlabel("وضعیت دیابت")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("bmi_diabetes.png")
plt.show()

# === نمودار ۴: ماتریس همبستگی ===
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("ماتریس همبستگی متغیرها", fontsize=14)
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

print("\n✅ همه نمودارها ساخته و ذخیره شدن!")