import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Titanic-Dataset.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== SUMMARY ==========")
print(df.describe())

# -----------------------------
# Visualization 1: Survival Count
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("images/survival_count.png")
plt.show()

# -----------------------------
# Visualization 2: Passenger Gender
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.savefig("images/gender_distribution.png")
plt.show()

# -----------------------------
# Visualization 3: Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("images/age_distribution.png")
plt.show()

# -----------------------------
# Visualization 4: Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

print("\nEDA Completed Successfully!")