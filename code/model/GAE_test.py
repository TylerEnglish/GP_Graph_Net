
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from GAE import GAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data = pickle.load(open('./data/dataset.pkl', 'rb'))

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)
gnn = GAE()
gnn = torch.load('./data/gnn_model.pt', map_location=device)
gnn.eval()

# Load test data
test_loader = DataLoader(test_data, batch_size=64)

'''

'''
