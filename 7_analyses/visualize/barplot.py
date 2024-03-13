import matplotlib.pyplot as plt
import numpy as np

# Define the data
sofa9 = [0.616, 0.200, 0.850, 0.714]
trigger = [0.810, 0.130, 0.933, 0.867]
sequence = [0.788, 0.190, 0.888, 0.835]
groups = ['TPR', 'FPRatio', 'PPV', 'F1']

# Setting up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Bar positions
barWidth = 0.15
r1 = np.arange(len(groups))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Plotting
ax.bar(r1, sofa9, color=(25/255, 49/255, 52/255), width=barWidth, edgecolor='grey', label='pivoted SOFA, threshold=9')
ax.bar(r2, trigger, color=(54/255, 104/255, 107/255), width=barWidth, edgecolor='grey', label='Trigger-based COT, maxFPR=0.15')
ax.bar(r3, sequence, color=(76/255, 164/255, 162/255), width=barWidth, edgecolor='grey', label='Sequence-based COT, maxFPR=0.40')

# Add some aesthetics
ax.set_xticks([r + barWidth for r in range(len(groups))])
ax.set_xticklabels(groups)

# Creating legend & title
ax.legend()
ax.set_title('Comparison of Online Prediction Metrics')

# Show plot
plt.show()