import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE

with open('./data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
# Define the graph encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Define the GAE model
class GAEModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GAEModel, self).__init__()
        self.encoder = Encoder(num_features, hidden_channels)
        self.gae = GAE(self.encoder)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.gae(x, edge_index)
        return z

# Train the GAE model
model = GAEModel(num_features=dataset.num_features, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

def train():
    model.train()
    optimizer.zero_grad()
    z = model(data)
    loss = criterion(z, data.x)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(100):
    loss = train()
    print('Epoch {:03d}, Loss: {:.4f}'.format(epoch, loss))