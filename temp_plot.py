import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Step 1: Input the data from your log files ---

# Class names in the order they appear in the logs
class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# IoU scores from the final epoch of each run
baseline_iou = [
    0.9632, 0.7279, 0.8651, 0.4195, 0.3934, 0.3164, 0.3008, 0.4586, 0.8687, 
    0.5440, 0.9056, 0.5696, 0.3205, 0.8755, 0.4765, 0.4794, 0.2577, 0.2789, 0.5483
]

advanced_iou = [
    0.9614, 0.7122, 0.8609, 0.4062, 0.3669, 0.3286, 0.3656, 0.5045, 0.8650, 
    0.4960, 0.9094, 0.5860, 0.3633, 0.8748, 0.4437, 0.5261, 0.3395, 0.3270, 0.5514
]

# Overall Mean IoU
baseline_miou = 0.5563
advanced_miou = 0.5678

# --- Step 2: Create a Pandas DataFrame for easier plotting ---
data = {
    'Class': class_names,
    'Baseline IoU': baseline_iou,
    'Advanced IoU': advanced_iou
}
df = pd.DataFrame(data)

# Sort by the performance gain to make the plot more insightful
df['Gain'] = df['Advanced IoU'] - df['Baseline IoU']
df_sorted = df.sort_values(by='Gain', ascending=False)


# --- Step 3: Create the plot ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(df_sorted['Class']))  # the label locations
width = 0.35  # the width of the bars

# Plotting the bars
rects1 = ax.bar(x - width/2, df_sorted['Baseline IoU'], width, label=f'Baseline (mIoU: {baseline_miou:.3f})', color='cornflowerblue')
rects2 = ax.bar(x + width/2, df_sorted['Advanced IoU'], width, label=f'Advanced (mIoU: {advanced_miou:.3f})', color='orangered')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Intersection-over-Union (IoU)', fontsize=14)
ax.set_title('Performance Comparison: Baseline vs. Advanced Method', fontsize=16, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_sorted['Class'], rotation=45, ha="right")
ax.legend(fontsize=12)
ax.set_ylim(0, 1.05) # Set y-axis limit from 0 to 1 for IoU

# Add a horizontal line at 0 for reference
ax.axhline(0, color='grey', linewidth=0.8)

# Add value labels on top of the bars (optional, can be cluttered)
# ax.bar_label(rects1, padding=3, fmt='%.2f', rotation=90, size=8)
# ax.bar_label(rects2, padding=3, fmt='%.2f', rotation=90, size=8)

fig.tight_layout() # Adjust layout to make room for rotated x-axis labels

# Save the plot to a file
plt.savefig("iou_comparison_chart.png", dpi=300)

# Show the plot
plt.show()