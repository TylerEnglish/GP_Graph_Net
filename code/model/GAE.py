import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, z):
        z = self.lin(z)
        return z

class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        z = F.relu(z)
        x_hat = self.decoder(z)
        return x_hat

# Load data
data = pickle.load(open('./data/dataset.pkl', 'rb'))
gnn = GAE(Encoder(data.num_features, 16, 8), Decoder(8, data.num_features))
gnn = gnn.to(device)

# Train loop
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
for epoch in range(100):
    gnn.train()
    optimizer.zero_grad()
    x_hat = gnn(data.x, data.edge_index)
    loss = criterion(x_hat, data.x)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate the trained model
gnn.eval()
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
with torch.no_grad():
    x_hat = gnn(data.x, data.edge_index)
    print("Reconstructed data: ")
    print(x_hat)
