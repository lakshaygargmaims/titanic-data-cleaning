"""
Exploratory Data Analysis - Titanic (Cleaned) Dataset
Week 2 Task: EDA and Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

IMG = "/home/claude/titanic_eda/images"
SURV_COLORS = {0: "#e15759", 1: "#4e79a7"}     # Did not survive: red, Survived: blue
SURV_LABELS = {0: "Did Not Survive", 1: "Survived"}

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/Titanic_cleaned.csv")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# ---------------------------------------------------------------
# 2. DATA QUALITY OBSERVATIONS (fare capping / age imputation spikes)
# ---------------------------------------------------------------
fare_cap = df["Fare"].max()
n_capped = (df["Fare"] == fare_cap).sum()
print(f"\nPassengers at max fare ({fare_cap}): {n_capped}")

age_spike_25 = (df["Age"] == 25.0).sum()
age_spike_215 = (df["Age"] == 21.5).sum()
print(f"Passengers with Age == 25.0: {age_spike_25}")
print(f"Passengers with Age == 21.5: {age_spike_215}")
print(f"Unique age values: {df['Age'].nunique()} (out of {len(df)} rows)")

# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare_titles = df["Title"].value_counts()
rare = rare_titles[rare_titles < 10].index
df["Title"] = df["Title"].replace(rare, "Other")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["SurvivedLabel"] = df["Survived"].map(SURV_LABELS)

print("\nTitle counts after grouping:\n", df["Title"].value_counts())

# ---------------------------------------------------------------
# 4. SUMMARY STATISTICS
# ---------------------------------------------------------------
num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
desc = df[num_cols].describe().T
desc.to_csv("/home/claude/titanic_eda/summary_statistics.csv")
print("\nSaved summary statistics.")

# ---------------------------------------------------------------
# FIGURE 1: Overall survival count
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = df["Survived"].value_counts().sort_index()
colors = [SURV_COLORS[i] for i in counts.index]
labels = [SURV_LABELS[i] for i in counts.index]
bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.6)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5, str(int(b.get_height())),
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Figure 1: Survival Outcome Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Outcome")
ax.set_ylabel("Number of Passengers")
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig(f"{IMG}/fig1_survival_balance.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 2: Survival rate by sex
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
rate = df.groupby("Sex")["Survived"].mean().sort_values()
bars = ax.bar(rate.index, rate.values * 100, color=["#4e79a7", "#e15759"], edgecolor="black", linewidth=0.6)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Figure 2: Survival Rate by Sex", fontsize=13, fontweight="bold")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{IMG}/fig2_survival_by_sex.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 3: Survival rate by passenger class
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
rate = df.groupby("Pclass")["Survived"].mean()
bars = ax.bar(["1st", "2nd", "3rd"], rate.values * 100, color="#59a14f", edgecolor="black", linewidth=0.6)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Figure 3: Survival Rate by Passenger Class", fontsize=13, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{IMG}/fig3_survival_by_class.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 4: Survival rate by class and sex (grouped bar)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
grp = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack() * 100
grp.plot(kind="bar", ax=ax, color=["#e15759", "#4e79a7"], edgecolor="black", linewidth=0.6)
ax.set_title("Figure 4: Survival Rate by Class and Sex", fontsize=13, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate (%)")
ax.set_xticklabels(["1st", "2nd", "3rd"], rotation=0)
ax.legend(title="Sex")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(f"{IMG}/fig4_survival_class_sex.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 5: Age distribution by survival
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.histplot(data=df, x="Age", hue="SurvivedLabel", kde=True, element="step",
             palette={"Survived": "#4e79a7", "Did Not Survive": "#e15759"}, ax=ax)
ax.set_title("Figure 5: Age Distribution by Survival Outcome", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{IMG}/fig5_age_distribution.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 6: Fare distribution by survival (with note on capping)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.histplot(data=df, x="Fare", hue="SurvivedLabel", kde=True, element="step",
             palette={"Survived": "#4e79a7", "Did Not Survive": "#e15759"}, ax=ax)
ax.set_title("Figure 6: Fare Distribution by Survival Outcome", fontsize=13, fontweight="bold")
ax.set_xlabel("Fare")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{IMG}/fig6_fare_distribution.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 7: Boxplots - Age and Fare by survival
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(data=df, x="SurvivedLabel", y="Age", hue="SurvivedLabel", legend=False,
            palette={"Survived": "#4e79a7", "Did Not Survive": "#e15759"}, ax=axes[0])
axes[0].set_title("Age by Survival", fontsize=12)
axes[0].set_xlabel("")
sns.boxplot(data=df, x="SurvivedLabel", y="Fare", hue="SurvivedLabel", legend=False,
            palette={"Survived": "#4e79a7", "Did Not Survive": "#e15759"}, ax=axes[1])
axes[1].set_title("Fare by Survival", fontsize=12)
axes[1].set_xlabel("")
fig.suptitle("Figure 7: Age and Fare Distributions by Survival Outcome", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{IMG}/fig7_boxplots_age_fare.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# FIGURE 8: Survival rate by family size
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
fam_rate = df.groupby("FamilySize")["Survived"].mean() * 100
fam_count = df.groupby("FamilySize")["Survived"].count()
bars = ax.bar(fam_rate.index.astype(str), fam_rate.values, color="#f28e2b", edgecolor="black", linewidth=0.6)
for b, n in zip(bars, fam_count.values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"n={n}",
            ha="center", fontsize=8)
ax.set_title("Figure 8: Survival Rate by Family Size", fontsize=13, fontweight="bold")
ax.set_xlabel("Family Size (Self + Siblings/Spouses + Parents/Children)")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{IMG}/fig8_survival_by_familysize.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 9: Correlation heatmap of numeric features
# ---------------------------------------------------------------
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.4, cbar_kws={"label": "Pearson Correlation"}, ax=ax)
ax.set_title("Figure 9: Correlation Heatmap of Numeric Features", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{IMG}/fig9_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 10: Scatter - Age vs Fare, colored by survival, sized by class
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.scatterplot(data=df, x="Age", y="Fare", hue="SurvivedLabel", style="Pclass",
                 palette={"Survived": "#4e79a7", "Did Not Survive": "#e15759"},
                 alpha=0.7, s=50, ax=ax)
ax.set_title("Figure 10: Age vs. Fare, by Survival and Class", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Fare")
ax.legend(title="Outcome / Class", fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/fig10_scatter_age_fare.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 11: Survival rate by title (engineered feature)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
title_rate = df.groupby("Title")["Survived"].mean().sort_values(ascending=False) * 100
title_n = df.groupby("Title")["Survived"].count()
bars = ax.bar(title_rate.index, title_rate.values, color="#76b7b2", edgecolor="black", linewidth=0.6)
for b, n in zip(bars, title_n.reindex(title_rate.index).values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"n={n}", ha="center", fontsize=8)
ax.set_title("Figure 11: Survival Rate by Title (Extracted from Name)", fontsize=13, fontweight="bold")
ax.set_xlabel("Title")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{IMG}/fig11_survival_by_title.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 12: Outlier counts (IQR method) for numeric features
# ---------------------------------------------------------------
outlier_counts = {}
for col in ["Age", "SibSp", "Parch", "Fare", "FamilySize"]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()
outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(outlier_series.index, outlier_series.values, color="#af7aa1", edgecolor="black", linewidth=0.5)
ax.set_title("Figure 12: IQR-Based Outlier Counts per Feature", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Outlier Observations")
plt.tight_layout()
plt.savefig(f"{IMG}/fig12_outlier_counts.png")
plt.close()

print("\nAll figures generated successfully.")
print("\nOutlier counts:\n", outlier_series)

# ---------------------------------------------------------------
# Print key grouped stats for narrative use
# ---------------------------------------------------------------
print("\nSurvival by Sex:\n", df.groupby("Sex")["Survived"].agg(["mean", "count"]))
print("\nSurvival by Pclass:\n", df.groupby("Pclass")["Survived"].agg(["mean", "count"]))
print("\nSurvival by Class+Sex:\n", df.groupby(["Pclass", "Sex"])["Survived"].mean())
print("\nSurvival by IsAlone:\n", df.groupby("IsAlone")["Survived"].agg(["mean", "count"]))
print("\nAge by survival:\n", df.groupby("Survived")["Age"].agg(["mean", "std", "median"]))
print("\nFare by survival:\n", df.groupby("Survived")["Fare"].agg(["mean", "std", "median"]))
print("\nCorrelation with Survived:\n", corr["Survived"].sort_values(ascending=False))
