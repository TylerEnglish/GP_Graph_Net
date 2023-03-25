import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)

# Further split the training set into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=123)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

num_features = max([dataset.num_features for dataset in train_data])
gnn = GAE(Encoder(num_features, 16, 8), Decoder(8, num_features))
gnn = gnn.to(device)

# Train loop with validation
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
best_val_loss = float('inf')

train_losses, val_losses = [], []

for epoch in range(100):
    gnn.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
        loss = criterion(x_hat, batch.x.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    
    train_loss /= len(train_data)
    
    gnn.eval()
    val_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
            loss = criterion(x_hat, batch.x.to(device))
            val_loss += loss.item() * batch.num_graphs
    
    val_loss /= len(val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gnn.state_dict(), './data/gnn_model.pt')

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

gnn.load_state_dict(torch.load('./data/gnn_model.pt'))
gnn.eval()
test_loss = 0
for batch in test_loader:
    with torch.no_grad():
        x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
        loss = criterion(x_hat, batch.x.to(device))
        test_loss += loss.item() * batch.num_graphs
        
test_loss /= len(test_data)

print(f"Test Loss: {test_loss:.4f}")


plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./data/gae_loss.png')


# Save test data
with open('./data/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

# Save test data
with open('./data/train.pkl', 'wb') as f:
    pickle.dump(train_data, f)