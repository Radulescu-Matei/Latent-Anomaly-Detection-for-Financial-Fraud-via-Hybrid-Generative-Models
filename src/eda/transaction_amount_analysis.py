import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('src/eda/graphs', exist_ok=True)

df_creditcard = pd.read_csv('Datasets/creditcard.csv')
df_ieee = pd.read_csv('Datasets/train_transaction.csv')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
df_creditcard[df_creditcard['Class'] == 0]['Amount'].hist(bins=50, alpha=0.7, color='#2ecc71', ax=ax1, edgecolor='black')
ax1.set_xlabel('Transaction Amount', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('CreditCard : Normal Transactions', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
df_creditcard[df_creditcard['Class'] == 1]['Amount'].hist(bins=50, alpha=0.7, color='#e74c3c', ax=ax2, edgecolor='black')
ax2.set_xlabel('Transaction Amount', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('CreditCard : Fraud Transactions', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
df_ieee[df_ieee['isFraud'] == 0]['TransactionAmt'].hist(bins=50, alpha=0.7, color='#2ecc71', ax=ax3, edgecolor='black')
ax3.set_xlabel('Transaction Amount', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('IEEE Dataset - Normal Transactions', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
df_ieee[df_ieee['isFraud'] == 1]['TransactionAmt'].hist(bins=50, alpha=0.7, color='#e74c3c', ax=ax4, edgecolor='black')
ax4.set_xlabel('Transaction Amount', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('IEEE Dataset - Fraud Transactions', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

plt.suptitle('Transaction Amount Distribution: Normal vs Fraud', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('src/eda/graphs/amount_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
data_cc = [df_creditcard[df_creditcard['Class'] == 0]['Amount'], 
           df_creditcard[df_creditcard['Class'] == 1]['Amount']]
bp1 = ax1.boxplot(data_cc, labels=['Normal', 'Fraud'], patch_artist=True, widths=0.6)
bp1['boxes'][0].set_facecolor('#2ecc71')
bp1['boxes'][1].set_facecolor('#e74c3c')
ax1.set_ylabel('Transaction Amount', fontsize=12, fontweight='bold')
ax1.set_title('CreditCard : Amount Box Plot', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3, axis='y')

ax2 = axes[1]
data_ieee = [df_ieee[df_ieee['isFraud'] == 0]['TransactionAmt'], 
             df_ieee[df_ieee['isFraud'] == 1]['TransactionAmt']]
bp2 = ax2.boxplot(data_ieee, labels=['Normal', 'Fraud'], patch_artist=True, widths=0.6)
bp2['boxes'][0].set_facecolor('#2ecc71')
bp2['boxes'][1].set_facecolor('#e74c3c')
ax2.set_ylabel('Transaction Amount', fontsize=12, fontweight='bold')
ax2.set_title('IEEE Dataset: Amount Box Plot', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

plt.suptitle('Transaction Amount Box Plots: Normal vs Fraud', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('src/eda/graphs/amount_boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
stats_cc = df_creditcard.groupby('Class')['Amount'].describe()[['mean', '50%', 'std', 'min', 'max']]
stats_cc.plot(kind='bar', ax=ax1, color=['#3498db', '#e67e22', '#9b59b6', '#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Amount', fontsize=12, fontweight='bold')
ax1.set_title('CreditCard : Amount Statistics', fontsize=13, fontweight='bold')
ax1.set_xticklabels(['Normal', 'Fraud'], rotation=0)
ax1.legend(['Mean', 'Median', 'Std Dev', 'Min', 'Max'], fontsize=9)
ax1.grid(alpha=0.3, axis='y')

ax2 = axes[1]
stats_ieee = df_ieee.groupby('isFraud')['TransactionAmt'].describe()[['mean', '50%', 'std', 'min', 'max']]
stats_ieee.plot(kind='bar', ax=ax2, color=['#3498db', '#e67e22', '#9b59b6', '#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
ax2.set_xlabel('isFraud', fontsize=12, fontweight='bold')
ax2.set_ylabel('Amount', fontsize=12, fontweight='bold')
ax2.set_title('IEEE Dataset: Amount Statistics', fontsize=13, fontweight='bold')
ax2.set_xticklabels(['Normal', 'Fraud'], rotation=0)
ax2.legend(['Mean', 'Median', 'Std Dev', 'Min', 'Max'], fontsize=9)
ax2.grid(alpha=0.3, axis='y')

plt.suptitle('Transaction Amount Statistics Comparison', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('src/eda/graphs/amount_statistics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()