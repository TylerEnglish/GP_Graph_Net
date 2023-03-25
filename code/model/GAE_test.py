import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import pickle
from GAE_P import Encoder, Decoder, GAE
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load data
test_data = pickle.load(open('./data/test_data.pkl', 'rb'))
train_data = pickle.load(open('./data/train_data.pkl', 'rb'))

num_features = max([dataset.num_features for dataset in train_data])
gnn = GAE(Encoder(num_features, 16, 8), Decoder(8, num_features))
gnn = gnn.to(device)

gnn.load_state_dict(torch.load('./data/gnn_model.pt', map_location=device))
gnn.eval()

# Load test data

test_loader = DataLoader(test_data, batch_size=64)

# Generate the graph of the input and output of the GAE
with torch.no_grad():
    for batch in test_loader:
        x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
        break

G1 = nx.Graph()
for i, (u, v) in enumerate(batch.edge_index.T.tolist()):
    G1.add_edge(u, v, weight=batch.x[i].item())

G2 = nx.Graph()
for i, (u, v) in enumerate(batch.edge_index.T.tolist()):
    G2.add_edge(u, v, weight=x_hat[i].item())

pos = nx.spring_layout(G1, seed=42)

# Plot input graph
fig, ax = plt.subplots(figsize=(10, 10))
nx.draw_networkx(G1, pos, node_color=batch.x.tolist(), with_labels=False, ax=ax)
ax.set_title('Input Graph')

# Plot output graph
fig, ax = plt.subplots(figsize=(10, 10))
nx.draw_networkx(G2, pos, node_color=x_hat.tolist(), with_labels=False, ax=ax)
ax.set_title('Output Graph')

plt.show()
