import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_creditcard = pd.read_csv('Datasets/creditcard.csv')


os.makedirs('src/eda/graphs', exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

creditcard_counts = df_creditcard['Class'].value_counts().sort_index()
normal_count = creditcard_counts[0]
fraud_count = creditcard_counts[1]
fraud_pct = (fraud_count / len(df_creditcard)) * 100

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['Normal', 'Fraud'], 
              [normal_count, fraud_count],
              color=['#2ecc71', '#e74c3c'],
              alpha=0.8,
              edgecolor='black',
              linewidth=1.5)

ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
ax.set_title('Creditcard Dataset - Class Distribution', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)


ax.text(0, normal_count, f'{normal_count:,}\n({100-fraud_pct:.3f}%)', 
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.text(1, fraud_count, f'{fraud_count:,}\n({fraud_pct:.3f}%)', 
        ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('src/eda/graphs/creditcard_class_distribution_bar.png', dpi=300, bbox_inches='tight')
plt.close()


df_ieee = pd.read_csv('Datasets/train_transaction.csv')


os.makedirs('src/eda/graphs', exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

ieee_counts = df_ieee['isFraud'].value_counts().sort_index()
normal_count = ieee_counts[0]
fraud_count = ieee_counts[1]
fraud_pct = (fraud_count / len(df_ieee)) * 100

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['Normal', 'Fraud'], 
              [normal_count, fraud_count],
              color=['#2ecc71', '#e74c3c'],
              alpha=0.8,
              edgecolor='black',
              linewidth=1.5)

ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
ax.set_title('IEEE Dataset - Class Distribution', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)


ax.text(0, normal_count, f'{normal_count:,}\n({100-fraud_pct:.3f}%)', 
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.text(1, fraud_count, f'{fraud_count:,}\n({fraud_pct:.3f}%)', 
        ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('src/eda/graphs/IEEE_class_distribution_bar.png', dpi=300, bbox_inches='tight')
plt.close()



