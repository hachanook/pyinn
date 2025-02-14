import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

# Generate dummy data (replace these with actual numpy arrays)
ndata, features, classes = 1000, 20, 5  # Example sizes
X = np.random.rand(ndata, features).astype(np.float32)
Y = np.random.rand(ndata, classes).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Create a dataset
dataset = TensorDataset(X_tensor, Y_tensor)

# Define split ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
train_size = int(train_ratio * ndata)
val_size = int(val_ratio * ndata)
test_size = ndata - train_size - val_size  # Ensures the full dataset is used

# Shuffle and split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define DataLoaders
batch_size = 32  # Change this as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example usage: Print first batch shape
for X_batch, Y_batch in train_loader:
    print(X_batch.shape, Y_batch.shape)
    break  # Print only the first batch shape
