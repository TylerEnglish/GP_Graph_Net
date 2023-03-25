import pickle
from torch_geometric.data import DataLoader
import torch
from torch_geometric.datasets import ZINC

# Load the dataset
dataset = ZINC(root='data/ZINC')

# Save the dataset as a file
with open('./data/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)