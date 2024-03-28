import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import csv

dataset_path = 'music_genre_dataset.csv'

# Initialize lists to hold features and labels
features = []
labels = []

# Read the dataset
with open(dataset_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row if there is one
    for row in csvreader:
        # Assuming all but the last column are features
        # Convert feature columns to float
        features.append([float(val) for val in row[:-1]])
        # Last column is the label
        labels.append(row[-1])

# Convert lists to appropriate data structures for plotting
# For simplicity, this example just converts them to lists of x and y for plotting
# This assumes the first two columns are the two features you want to plot
x = [feature[0] for feature in features]  # Feature 1
y = [feature[1] for feature in features]  # Feature 2

# Plotting
plt.figure(figsize=(8, 6))

# Assuming you have a small number of unique labels and want to color-code them
unique_labels = list(set(labels))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    xi = [x[j] for j in range(len(x)) if labels[j] == label]
    yi = [y[j] for j in range(len(y)) if labels[j] == label]
    plt.scatter(xi, yi, color=colors[i], label=label)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dataset Plot')
plt.legend()
plt.show()



