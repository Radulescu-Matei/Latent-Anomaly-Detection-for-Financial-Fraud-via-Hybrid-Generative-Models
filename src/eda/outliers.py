from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('src/eda/graphs', exist_ok=True)
os.makedirs('src/eda/datafiles', exist_ok=True)

df_creditcard = pd.read_csv('Datasets/creditcard.csv')
numeric_features_cc = df_creditcard.select_dtypes(include=['float64', 'int64']).columns
numeric_features_cc = [f for f in numeric_features_cc if f != 'Class']

outlier_pct_cc = {}
for feature in numeric_features_cc:
    z_scores = np.abs(stats.zscore(df_creditcard[feature].dropna()))
    outliers_pct = (z_scores > 3).sum() / len(df_creditcard) * 100
    outlier_pct_cc[feature] = outliers_pct

outlier_series_cc = pd.Series(outlier_pct_cc).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 12))

y_pos = np.arange(len(outlier_series_cc))
bars = ax.barh(y_pos, outlier_series_cc, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(outlier_series_cc.index, fontsize=9)
ax.set_xlabel('Outlier Percentage (Z-score > 3)', fontsize=12, fontweight='bold')
ax.set_title(f'CreditCard Dataset: Outlier Percentage Per Feature\n({len(outlier_series_cc)} features)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('src/eda/graphs/creditcard_outlier_percentages.png', dpi=300, bbox_inches='tight')
plt.close()

with open('src/eda/datafiles/creditcard_outlier_percentages.txt', 'w') as f:
    f.write("CreditCard Dataset: Outlier Percentages (Z-score > 3)\n")
    f.write("\n\n")
    for feature, pct in outlier_series_cc.items():
        f.write(f"{feature:30s}: {pct:6.3f}%\n")
    f.write("\n")
    f.write(f"Average outlier percentage: {outlier_series_cc.mean():.3f}%\n")
    f.write("\n\n")

df_ieee = pd.read_csv('Datasets/train_transaction.csv')
numeric_features_ieee = df_ieee.select_dtypes(include=['float64', 'int64']).columns
numeric_features_ieee = [f for f in numeric_features_ieee if f not in ['isFraud', 'TransactionID']]

outlier_pct_ieee = {}
for feature in numeric_features_ieee:
    z_scores = np.abs(stats.zscore(df_ieee[feature].dropna()))
    outliers_pct = (z_scores > 3).sum() / len(df_ieee) * 100
    outlier_pct_ieee[feature] = outliers_pct

outlier_series_ieee = pd.Series(outlier_pct_ieee).sort_values(ascending=False)

top_30_ieee = outlier_series_ieee.head(30)

fig, ax = plt.subplots(figsize=(10, 12))

y_pos = np.arange(len(top_30_ieee))
bars = ax.barh(y_pos, top_30_ieee, color='#e67e22', alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_30_ieee.index, fontsize=9)
ax.set_xlabel('Outlier Percentage (Z-score > 3)', fontsize=12, fontweight='bold')
ax.set_title(f'IEEE Dataset - Top 30 Features by Outlier Percentage\n({len(outlier_series_ieee)} total features)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('src/eda/graphs/IEEE_outlier_percentages_top30.png', dpi=300, bbox_inches='tight')
plt.close()

with open('src/eda/datafiles/IEEE_outlier_percentages.txt', 'w') as f:
    f.write("IEEE Dataset: Outlier Percentages (Z-score > 3)\n")
    f.write("\n\n")
    for feature, pct in outlier_series_ieee.items():
        f.write(f"{feature:30s}: {pct:6.3f}%\n")
    f.write("\n")
    f.write(f"Average outlier percentage: {outlier_series_ieee.mean():.3f}%\n")


fig, ax = plt.subplots(figsize=(10, 6))

datasets = ['CreditCard\nDataset', 'IEEE-CIS\nDataset']
avg_outliers = [outlier_series_cc.mean(), outlier_series_ieee.mean()]
colors = ['#3498db', '#e67e22']

bars = ax.bar(datasets, avg_outliers, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Average Outlier Percentage', fontsize=12, fontweight='bold')
ax.set_title('Average Outlier Rate Comparison Between Datasets\n(Z-score > 3 threshold)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, pct in zip(bars, avg_outliers):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('src/eda/graphs/outlier_comparison_datasets.png', dpi=300, bbox_inches='tight')
plt.close()
