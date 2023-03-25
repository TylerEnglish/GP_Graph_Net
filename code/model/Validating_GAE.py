import pickle
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from GAE_P import myGAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = pickle.load(open('./data/dataset.pkl', 'rb'))
data = dataset[0]

# Draw graph
fig = plt.figure(figsize=(20, 15))
networkx_graph = to_networkx(data)
nx.draw_networkx(networkx_graph)
plt.show()

# Create GAE model and load pre-trained parameters
gae = myGAE(data.num_features, 16, data.num_features)
gae.load_state_dict(torch.load('./data/gae_model.pt'))

# Set model to evaluation mode
gae.eval()

# Apply the model to the input data without computing gradients
with torch.no_grad():
    z = gae.encode(data.x, data.edge_index)
    out = gae.decode(z, data.edge_index)
