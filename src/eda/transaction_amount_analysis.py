import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('src/eda/graphs', exist_ok=True)
os.makedirs('src/eda/datafiles', exist_ok=True)

df_creditcard = pd.read_csv('Datasets/creditcard.csv')
df_ieee = pd.read_csv('Datasets/train_transaction.csv')

amount_bins = [0, 50, 100, 200, 500, 1000, 5000]
amount_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000-5000']

df_creditcard['AmountCategory'] = pd.cut(df_creditcard['Amount'], bins=amount_bins, labels=amount_labels)
df_ieee['AmountCategory'] = pd.cut(df_ieee['TransactionAmt'], bins=amount_bins, labels=amount_labels)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax1 = axes[0]
all_cc_counts = df_creditcard['AmountCategory'].value_counts().sort_index()

bars1 = ax1.bar(range(len(all_cc_counts)), all_cc_counts, color='#3498db', alpha=0.8, edgecolor='black')

ax1.set_xlabel('Amount Range', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
ax1.set_title('CreditCard Dataset: Amount Distribution', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(all_cc_counts)))
ax1.set_xticklabels(all_cc_counts.index, rotation=45, ha='right')
ax1.grid(alpha=0.3, axis='y')

for bar, count in zip(bars1, all_cc_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2 = axes[1]
all_ieee_counts = df_ieee['AmountCategory'].value_counts().sort_index()

bars2 = ax2.bar(range(len(all_ieee_counts)), all_ieee_counts, color='#e67e22', alpha=0.8, edgecolor='black')

ax2.set_xlabel('Amount Range', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
ax2.set_title('IEEE Dataset: Amount Distribution', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(all_ieee_counts)))
ax2.set_xticklabels(all_ieee_counts.index, rotation=45, ha='right')
ax2.grid(alpha=0.3, axis='y')

for bar, count in zip(bars2, all_ieee_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Transaction Amount Distribution', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('src/eda/graphs/amount_ranges_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax1 = axes[0]
fraud_cc = df_creditcard[df_creditcard['Class'] == 1]
ax1.scatter(range(len(fraud_cc)), fraud_cc['Amount'], c='#e74c3c', alpha=0.7, s=20, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Fraud Transaction Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Transaction Amount', fontsize=12, fontweight='bold')
ax1.set_title('CreditCard Dataset: Fraud Transactions', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

ax2 = axes[1]
fraud_ieee = df_ieee[df_ieee['isFraud'] == 1]
ax2.scatter(range(len(fraud_ieee)), fraud_ieee['TransactionAmt'], c='#e74c3c', alpha=0.7, s=20, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Fraud Transaction Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('Transaction Amount', fontsize=12, fontweight='bold')
ax2.set_title('IEEE Dataset: Fraud Transactions', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

plt.suptitle('Transaction Amount: Fraud Transactions', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('src/eda/graphs/amount_fraud_transactions.png', dpi=300, bbox_inches='tight')
plt.close()

stats_cc_normal = df_creditcard[df_creditcard['Class'] == 0]['Amount'].describe()
stats_cc_fraud = df_creditcard[df_creditcard['Class'] == 1]['Amount'].describe()

stats_ieee_normal = df_ieee[df_ieee['isFraud'] == 0]['TransactionAmt'].describe()
stats_ieee_fraud = df_ieee[df_ieee['isFraud'] == 1]['TransactionAmt'].describe()

with open('src/eda/datafiles/amount_statistics.txt', 'w') as f:
    f.write("Transaction Amount Statistics\n")
    f.write("\n\n")
    
    f.write("CREDITCARD DATASET:\n")
    f.write("\n")
    f.write("Normal Transactions:\n")
    f.write(f"  Count:    {stats_cc_normal['count']:,.0f}\n")
    f.write(f"  Mean:     {stats_cc_normal['mean']:,.2f}\n")
    f.write(f"  Std Dev:  {stats_cc_normal['std']:,.2f}\n")
    f.write(f"  Min:      {stats_cc_normal['min']:,.2f}\n")
    f.write(f"  25%:      {stats_cc_normal['25%']:,.2f}\n")
    f.write(f"  Median:   {stats_cc_normal['50%']:,.2f}\n")
    f.write(f"  75%:      {stats_cc_normal['75%']:,.2f}\n")
    f.write(f"  Max:      {stats_cc_normal['max']:,.2f}\n")
    f.write("\n")
    f.write("Fraud Transactions:\n")
    f.write(f"  Count:    {stats_cc_fraud['count']:,.0f}\n")
    f.write(f"  Mean:     {stats_cc_fraud['mean']:,.2f}\n")
    f.write(f"  Std Dev:  {stats_cc_fraud['std']:,.2f}\n")
    f.write(f"  Min:      {stats_cc_fraud['min']:,.2f}\n")
    f.write(f"  25%:      {stats_cc_fraud['25%']:,.2f}\n")
    f.write(f"  Median:   {stats_cc_fraud['50%']:,.2f}\n")
    f.write(f"  75%:      {stats_cc_fraud['75%']:,.2f}\n")
    f.write(f"  Max:      {stats_cc_fraud['max']:,.2f}\n")
    f.write("\n")
    
    f.write("\n\n")
    
    f.write("IEEE DATASET:\n")
    f.write("\n")
    f.write("Normal Transactions:\n")
    f.write(f"  Count:    {stats_ieee_normal['count']:,.0f}\n")
    f.write(f"  Mean:     {stats_ieee_normal['mean']:,.2f}\n")
    f.write(f"  Std Dev:  {stats_ieee_normal['std']:,.2f}\n")
    f.write(f"  Min:      {stats_ieee_normal['min']:,.2f}\n")
    f.write(f"  25%:      {stats_ieee_normal['25%']:,.2f}\n")
    f.write(f"  Median:   {stats_ieee_normal['50%']:,.2f}\n")
    f.write(f"  75%:      {stats_ieee_normal['75%']:,.2f}\n")
    f.write(f"  Max:      {stats_ieee_normal['max']:,.2f}\n")
    f.write("\n")
    f.write("Fraud Transactions:\n")
    f.write(f"  Count:    {stats_ieee_fraud['count']:,.0f}\n")
    f.write(f"  Mean:     {stats_ieee_fraud['mean']:,.2f}\n")
    f.write(f"  Std Dev:  {stats_ieee_fraud['std']:,.2f}\n")
    f.write(f"  Min:      {stats_ieee_fraud['min']:,.2f}\n")
    f.write(f"  25%:      {stats_ieee_fraud['25%']:,.2f}\n")
    f.write(f"  Median:   {stats_ieee_fraud['50%']:,.2f}\n")
    f.write(f"  75%:      {stats_ieee_fraud['75%']:,.2f}\n")
    f.write(f"  Max:      {stats_ieee_fraud['max']:,.2f}\n")