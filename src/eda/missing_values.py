import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('src/eda/graphs', exist_ok=True)
os.makedirs('src/eda/datafiles', exist_ok=True)

df_ieee_transaction = pd.read_csv('Datasets/train_transaction.csv')
df_ieee_identity = pd.read_csv('Datasets/train_identity.csv')

transaction_ids_all = set(df_ieee_transaction['TransactionID'])
identity_ids = set(df_ieee_identity['TransactionID'])

matched_ids = transaction_ids_all & identity_ids
unmatched_ids = transaction_ids_all - identity_ids

matched_count = len(matched_ids)
unmatched_count = len(unmatched_ids)
matched_pct = (matched_count / len(transaction_ids_all)) * 100
unmatched_pct = (unmatched_count / len(transaction_ids_all)) * 100

df_ieee = df_ieee_transaction.merge(df_ieee_identity, on='TransactionID', how='left')

if 'TransactionID' in df_ieee.columns:
    df_ieee = df_ieee.drop(columns=['TransactionID'])

feature_cols = [col for col in df_ieee.columns if col != 'isFraud']
df_features = df_ieee[feature_cols]

missing_counts = df_features.isnull().sum()
missing_pct = (missing_counts / len(df_features) * 100).sort_values(ascending=False)

missing_pct_nonzero = missing_pct[missing_pct > 0]

top_30_missing = missing_pct_nonzero.head(30)

fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(top_30_missing))
bars = ax.barh(y_pos, top_30_missing, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_30_missing.index, fontsize=9)
ax.set_xlabel('Missing Values (%)', fontsize=12, fontweight='bold')
ax.set_title(f'IEEE Dataset: Top 30 Features with Missing Values\n({len(missing_pct_nonzero)} features have missing values)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

for i, (bar, pct) in enumerate(zip(bars, top_30_missing)):
    ax.text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('src/eda/graphs/IEEE_missing_values_top30.png', dpi=300, bbox_inches='tight')
plt.close()

missing_summary = {
    'Very High (>75%)': (missing_pct > 75).sum(),
    'High (50-75%)': ((missing_pct > 50) & (missing_pct <= 75)).sum(),
    'Moderate (25-50%)': ((missing_pct > 25) & (missing_pct <= 50)).sum(),
    'Low (5-25%)': ((missing_pct > 5) & (missing_pct <= 25)).sum(),
    'Very Low (0-5%)': ((missing_pct > 0) & (missing_pct <= 5)).sum(),
    'None (0%)': (missing_pct == 0).sum()
}

fig, ax = plt.subplots(figsize=(12, 6))

colors_summary = ['#8b0000', '#e74c3c', '#f39c12', '#f1c40f', '#95a5a6', '#2ecc71']
bars = ax.bar(missing_summary.keys(), missing_summary.values(), color=colors_summary, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax.set_title(f'IEEE Dataset: Missing Values Distribution\n({len(feature_cols)} total features)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=0)

for bar, count in zip(bars, missing_summary.values()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/len(feature_cols)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('src/eda/graphs/IEEE_missing_values_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

with open('src/eda/datafiles/IEEE_missing_values.txt', 'w') as f:
    f.write("IEEE Dataset:Missing Values Analysis\n")
    f.write("\n\n")
    f.write("IDENTITY DATA MATCH:\n")
    f.write("\n")
    f.write(f"Total transactions: {len(transaction_ids_all):,}\n")
    f.write(f"Transactions with identity data: {matched_count:,} ({matched_pct:.2f}%)\n")
    f.write(f"Transactions without identity data: {unmatched_count:,} ({unmatched_pct:.2f}%)\n")
    f.write("\n")
    f.write("MISSING VALUES BY FEATURE:\n")
    f.write("\n")
    for feature, pct in missing_pct.items():
        count = missing_counts[feature]
        f.write(f"{feature:30s}: {count:8d} ({pct:6.2f}%)\n")
    f.write("\n\n")
    f.write(f"Total features (excluding target): {len(feature_cols)}\n")
    f.write(f"Features with missing values: {len(missing_pct_nonzero)}\n")
    f.write(f"Features with no missing values: {(missing_pct == 0).sum()}\n")
    f.write(f"Average missing percentage: {missing_pct.mean():.2f}%\n")