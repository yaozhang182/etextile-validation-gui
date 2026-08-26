import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_name = './angle_results/train_angle.npy'
train_angle = np.load(file_name)

# Create a figure with 3 subplots, sharing the x-axis
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Define some colors and labels for clarity
colors = ['r', 'g', 'b']
labels = ['Angle 1 (Column 0)', 'Angle 2 (Column 1)', 'Angle 3 (Column 2)']

# Loop through each column and plot on its respective subplot
for i in range(3):
    axs[i].plot(train_angle[:, i], color=colors[i], label=labels[i])
    axs[i].set_ylabel('Angle Value')
    axs[i].set_title(f'Angle {i+1} Visualization')
    axs[i].legend(loc='upper right')
    axs[i].grid(True)

# Set the x-axis label on the bottom plot only
axs[-1].set_xlabel('Frame')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()