import matplotlib.pyplot as plt
import numpy as np

# Your hard-earned data
m_values = ['16', '32', '64']
pro_auc = [0.2201, 0.2932, 0.9778]
latency = [577.07, 599.54, 600.02]

# Set up the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot PRO-AUC on the primary y-axis (Left)
color1 = 'tab:blue'
ax1.set_xlabel('FAISS Subquantizers (M)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (PRO-AUC)', color=color1, fontsize=12, fontweight='bold')
line1 = ax1.plot(m_values, pro_auc, color=color1, marker='o', markersize=8, linewidth=3, label='Accuracy (PRO-AUC)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.1)
ax1.grid(True, linestyle='--', alpha=0.6)

# Create a secondary y-axis for Latency (Right)
ax2 = ax1.twinx()  
color2 = 'tab:red'
ax2.set_ylabel('Edge Latency (ms)', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(m_values, latency, color=color2, marker='s', markersize=8, linewidth=3, linestyle='--', label='Inference Latency (ms)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(550, 650) # Keep scale tight to show it's mostly flat

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=11)

# Title and Layout
plt.title('RQ1: The Quantization Trade-off (Category: Screw)', fontsize=14, fontweight='bold')
fig.tight_layout()

# Save the graph
plt.savefig('rq1_tradeoff_graph.png', dpi=300)
print("Graph saved successfully as rq1_tradeoff_graph.png!")