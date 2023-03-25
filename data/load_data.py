import pickle
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
# Load the dataset
dataset = QM9(root='data/QM9')

# Save the dataset as a file
with open('./data/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)