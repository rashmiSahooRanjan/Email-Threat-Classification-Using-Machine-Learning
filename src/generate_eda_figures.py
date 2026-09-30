"""
Script to generate all required EDA figures and save them to outputs/figures/
"""

import os
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/figures", exist_ok=True)
sns.set_theme(style="whitegrid", font="sans-serif")

# Load cached engineered features
feat_path = "outputs/reports/engineered_features.csv"
if not os.path.exists(feat_path):
    raise FileNotFoundError(f"Missing {feat_path}")

df = pd.read_csv(feat_path)
print(f"Loaded {len(df):,} records for EDA plotting.")

# 1. 01_class_distribution.png
plt.figure(figsize=(7, 5))
class_counts = df["label"].value_counts()
colors = ["#2b5c8f", "#d9534f"]
bars = plt.bar(["Routine / Legitimate (0)", "Threatening / Phishing (1)"], class_counts.values, color=colors, width=0.55)
for bar in bars:
    yval = bar.get_height()
    pct = yval / len(df) * 100
    plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, weight="bold")
plt.title("Class Distribution in Cleaned Email Dataset (N = 82,391)", fontsize=13, weight="bold", pad=12)
plt.ylabel("Number of Emails", fontsize=11)
plt.ylim(0, max(class_counts.values) * 1.15)
plt.tight_layout()
plt.savefig("outputs/figures/01_class_distribution.png", dpi=300)
plt.close()
print("Saved 01_class_distribution.png")

# 2. 02_missing_values.png
plt.figure(figsize=(8, 4))
# Source datasets missing subject counts
missing_data = pd.Series({
    "Enron.csv": 198,
    "Ling.csv": 62,
    "Nigerian_Fraud.csv": 39,
    "CEAS_08.csv": 28,
    "SpamAssasin.csv": 16,
    "Nazario.csv": 4,
    "Cleaned_Features": 0
})
bars = plt.bar(missing_data.index, missing_data.values, color="#e67e22", width=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 3, f"{int(yval)}", ha="center", va="bottom", fontsize=9, weight="bold")
plt.title("Missing Subject Headers Across Source Corpora vs. Cleaned Dataset", fontsize=12, weight="bold", pad=12)
plt.ylabel("Missing Values Count", fontsize=10)
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.ylim(0, max(missing_data.values) * 1.18)
plt.tight_layout()
plt.savefig("outputs/figures/02_missing_values.png", dpi=300)
plt.close()
print("Saved 02_missing_values.png")

# 3. 03_message_length_distribution.png
plt.figure(figsize=(9, 5))
# Clip extreme outliers for visual clarity in histogram
clipped_len = np.clip(df["message_length"], 0, 10000)
sns.histplot(data=df.assign(clipped_len=clipped_len), x="clipped_len", hue="label", palette={0: "#2b5c8f", 1: "#d9534f"}, bins=50, kde=True, common_norm=False)
plt.title("Distribution of Email Message Length by Class (Clipped at 10k chars)", fontsize=13, weight="bold", pad=12)
plt.xlabel("Message Length (Characters)", fontsize=11)
plt.ylabel("Density / Frequency", fontsize=11)
plt.legend(["Threatening (1)", "Routine (0)"], frameon=True)
plt.tight_layout()
plt.savefig("outputs/figures/03_message_length_distribution.png", dpi=300)
plt.close()
print("Saved 03_message_length_distribution.png")

# 4. 04_url_count_distribution.png
plt.figure(figsize=(9, 5))
clipped_url = np.clip(df["url_count"], 0, 15)
sns.countplot(data=df.assign(clipped_url=clipped_url), x="clipped_url", hue="label", palette={0: "#2b5c8f", 1: "#d9534f"})
plt.title("Distribution of URL Count per Email by Class (Clipped at 15 URLs)", fontsize=13, weight="bold", pad=12)
plt.xlabel("Detected Hyperlinks Count", fontsize=11)
plt.ylabel("Count of Emails", fontsize=11)
plt.legend(["Routine (0)", "Threatening (1)"], frameon=True)
plt.tight_layout()
plt.savefig("outputs/figures/04_url_count_distribution.png", dpi=300)
plt.close()
print("Saved 04_url_count_distribution.png")

# 5. 05_feature_boxplots.png
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
features_to_box = [
    ("suspicious_keyword_count", "Suspicious Keywords", 15),
    ("urgency_keyword_count", "Urgency Keywords", 8),
    ("uppercase_ratio", "Uppercase Ratio", 1.0),
    ("special_char_count", "Special Characters", 500)
]
for idx, (col, title, clip_val) in enumerate(features_to_box):
    ax = axes[idx // 2, idx % 2]
    df_temp = df.copy()
    df_temp["label_name"] = df_temp["label"].map({0: "Routine (0)", 1: "Threat (1)"})
    sns.boxplot(data=df_temp, x="label_name", y=col, ax=ax, hue="label_name", palette=["#90caf9", "#ef9a9a"], legend=False)
    ax.set_title(f"{title} (by Class)", fontsize=11, weight="bold")
    ax.set_xlabel("Email Class", fontsize=10)
    ax.set_ylabel(col, fontsize=10)
plt.suptitle("Boxplot Distributions of Key Structural Indicators", fontsize=14, weight="bold", y=0.98)
plt.tight_layout()
plt.savefig("outputs/figures/05_feature_boxplots.png", dpi=300)
plt.close()
print("Saved 05_feature_boxplots.png")

# 6. 06_correlation_heatmap.png
plt.figure(figsize=(12, 10))
sample_cols = [
    "label", "message_length", "word_count", "url_count", "has_url",
    "suspicious_keyword_count", "urgency_keyword_count", "exclamation_count",
    "uppercase_ratio", "digit_ratio", "special_char_count", "currency_symbol_count",
    "html_indicator", "attachment_indicator"
]
corr = df[sample_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="vlag", annot=True, fmt=".2f", vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap: Observable Characteristics vs. Threat Label", fontsize=13, weight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/figures/06_correlation_heatmap.png", dpi=300)
plt.close()
print("Saved 06_correlation_heatmap.png")

# 7. 07_feature_target_relationship.png
plt.figure(figsize=(10, 5))
target_corr = corr["label"].drop("label").sort_values(ascending=False)
colors = ["#d9534f" if v > 0 else "#2b5c8f" for v in target_corr.values]
bars = plt.barh(target_corr.index, target_corr.values, color=colors, alpha=0.85)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Pearson Correlation of Observable Characteristics with Threat Class (Label = 1)", fontsize=13, weight="bold", pad=12)
plt.xlabel("Correlation Coefficient with Threat Label", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/figures/07_feature_target_relationship.png", dpi=300)
plt.close()
print("Saved 07_feature_target_relationship.png")

print("All 7 EDA figures successfully generated!")
