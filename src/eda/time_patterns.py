import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('src/eda/graphs', exist_ok=True)

df_creditcard = pd.read_csv('Datasets/creditcard.csv')

time_bins = np.linspace(df_creditcard['Time'].min(), df_creditcard['Time'].max(), 49)
df_creditcard['TimeBin'] = pd.cut(df_creditcard['Time'], bins=time_bins)

fraud_by_bin = df_creditcard[df_creditcard['Class'] == 1].groupby('TimeBin').size()
normal_by_bin = df_creditcard[df_creditcard['Class'] == 0].groupby('TimeBin').size()

bin_centers = [(interval.left + interval.right) / 2 for interval in fraud_by_bin.index]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
ax1.plot(bin_centers, fraud_by_bin.values, color='#e74c3c', linewidth=2, marker='o', markersize=4, label='Fraud')
ax1.plot(bin_centers, normal_by_bin.values, color='#2ecc71', linewidth=2, marker='o', markersize=4, label='Normal')
ax1.set_xlabel('Time (moment)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
ax1.set_title('CreditCard Dataset": Transaction Distribution Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

ax2 = axes[1]
fraud_rate_by_bin = df_creditcard.groupby('TimeBin')['Class'].mean() * 100
ax2.plot(bin_centers, fraud_rate_by_bin.values, color='#e67e22', linewidth=2, marker='o', markersize=4)
ax2.set_xlabel('Time (moment)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Fraud Rate', fontsize=12, fontweight='bold')
ax2.set_title('CreditCard Dataset: Fraud Rate Over Time', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('src/eda/graphs/creditcard_time_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
