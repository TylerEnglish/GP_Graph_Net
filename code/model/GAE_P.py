import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import ZINC


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.to(torch.float)  # Cast tensor to Float
        edge_index = edge_index.to(torch.float)  # Cast tensor to Float
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
dataset = ZINC('./data/ZINC').shuffle()
data = dataset[0].to(device)

# Define training function
def train(model, optimizer, loader, criterion):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.x)
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


# Define testing function
def test(model, loader, criterion):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.x.to(torch.float)) # Cast the data.x to Float
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


# Split data into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.1)

# Define data loaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# Define model, optimizer, and loss function
model = GAE(Encoder(dataset.num_features, 64, 32), Decoder(32, dataset.num_features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Train the model and plot training and validation losses
num_epochs = 100
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss = train(model, optimizer, train_loader, criterion)
    test_loss = test(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print('Epoch: {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the input and output graphs
data = dataset[0]
x, edge_index = data.x.to(device), data.edge_index.to(device)
model.eval()
with torch.no_grad():
    z = model.encoder(x, edge_index)
    z = F.relu(z)
    x_hat = model.decoder(z)

# Convert the input and output graphs to NetworkX format for plotting
G_input = to_networkx(Data(x=x, edge_index=edge_index))
G_output = to_networkx(Data(x=x_hat.cpu(), edge_index=edge_index.cpu()))

# Plot the input and output graphs side-by-side
fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
axes[0].set_title('Input Graph')
axes[1].set_title('Output Graph')
pos = nx.spring_layout(G_input)
nx.draw(G_input, pos, with_labels=True, ax=axes[0])
nx.draw(G_output, pos, with_labels=True, ax=axes[1])
plt.show()
