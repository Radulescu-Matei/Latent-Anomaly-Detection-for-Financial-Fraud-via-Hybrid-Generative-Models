import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
import numpy as np
from scipy.stats import chi2_contingency

os.makedirs('src/eda/datafiles', exist_ok=True)

df_creditcard = pd.read_csv('Datasets/creditcard.csv')
fig, ax = plt.subplots(figsize=(10, 12))

class_correlation = df_creditcard.corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)

colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in class_correlation]
y_pos = np.arange(len(class_correlation))

bars = ax.barh(y_pos, class_correlation, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(class_correlation.index, fontsize=9)
ax.set_xlabel('Correlation with Fraud', fontsize=12, fontweight='bold')
ax.set_title('All Features Ranked by Correlation with Fraud', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

legend_elements = [
    Patch(facecolor='#e74c3c', label='Positive (higher value = more likely fraud)'),
    Patch(facecolor='#2ecc71', label='Negative (higher value = more likely normal)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('src/eda/graphs/creditcard_features_correlation_ranked.png', dpi=300, bbox_inches='tight')
plt.close()

with open('src/eda/datafiles/creditcard_correlations.txt', 'w') as f:
    f.write("Credit Card Dataset: Feature Correlations with Fraud\n")
    f.write("\n\n")
    for feature, corr in class_correlation.items():
        f.write(f"{feature:30s}: {corr:+7.4f}\n")
    f.write("\n\n")

df_ieee_transaction = pd.read_csv('Datasets/train_transaction.csv')
df_ieee_identity = pd.read_csv('Datasets/train_identity.csv')

df_ieee = df_ieee_transaction.merge(df_ieee_identity, on='TransactionID', how='left')

if 'TransactionID' in df_ieee.columns:
    df_ieee = df_ieee.drop(columns=['TransactionID'])

numeric_cols = df_ieee.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
if 'isFraud' in numeric_cols:
    numeric_cols.remove('isFraud')
df_ieee_numeric = df_ieee[numeric_cols + ['isFraud']]
numeric_correlation = df_ieee_numeric.corr()['isFraud'].drop('isFraud')

categorical_cols = df_ieee.select_dtypes(include=['object', 'category']).columns.tolist()
if len(categorical_cols) > 0:
    df_categorical = df_ieee[categorical_cols + ['isFraud']].copy()
    df_categorical_encoded = df_categorical.apply(lambda x: x.factorize()[0])
    categorical_correlation = df_categorical_encoded.corr()['isFraud'].drop('isFraud')
    all_correlation = pd.concat([numeric_correlation, categorical_correlation])
else:
    all_correlation = numeric_correlation

fig, ax = plt.subplots(figsize=(14, 8))

summary_stats = {
    'Strong Positive\n(r > 0.3)': (all_correlation > 0.3).sum(),
    'Moderate Positive\n(0.1 < r ≤ 0.3)': ((all_correlation > 0.1) & (all_correlation <= 0.3)).sum(),
    'Weak Positive\n(0 < r ≤ 0.1)': ((all_correlation > 0) & (all_correlation <= 0.1)).sum(),
    'Zero\n(r = 0)': (all_correlation == 0).sum(),
    'Weak Negative\n(-0.1 ≤ r < 0)': ((all_correlation >= -0.1) & (all_correlation < 0)).sum(),
    'Moderate Negative\n(-0.3 ≤ r < -0.1)': ((all_correlation >= -0.3) & (all_correlation < -0.1)).sum(),
    'Strong Negative\n(r < -0.3)': (all_correlation < -0.3).sum()
}

colors_summary = ['#c0392b', '#e74c3c', '#ec7063', '#95a5a6', '#85c1e9', '#3498db', '#21618c']
bars = ax.bar(summary_stats.keys(), summary_stats.values(), color=colors_summary, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax.set_title(f'IEEE Dataset - Feature Correlation Distribution by Strength & Direction\n({len(all_correlation)} total features)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', labelsize=10)

for bar, count in zip(bars, summary_stats.values()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/len(all_correlation)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('src/eda/graphs/IEEE_all_features_correlation_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

all_correlation_sorted = all_correlation.sort_values(key=abs, ascending=False)

with open('src/eda/datafiles/IEEE_correlations.txt', 'w') as f:
    f.write("IEEE Dataset:Feature Correlations with Fraud\n")
    f.write("\n\n")
    for feature, corr in all_correlation_sorted.items():
        f.write(f"{feature:30s}: {corr:+7.4f}\n")
    f.write("\n\n")
