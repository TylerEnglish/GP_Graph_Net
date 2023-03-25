import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myGAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(myGAE, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


data = pickle.load(open('./data/dataset.pkl', 'rb'))
print(data)
# get subset of data
data = data[:10000]

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

batch_size = 64

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Define the model
gae = myGAE(data.num_features, 16, data.num_features)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gae = gae.to(device)

# Define the loss function
criterion = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)

# Set the model to training mode
gae.train()

# Define the number of training epochs
num_epochs = 10

gae = gae.to(device)

train_losses, val_losses = [], []

# Train loop with validation
optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
best_val_loss = float('inf')

for epoch in range(num_epochs):
    gae.train()  # set the model to train mode
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = gae(batch.x.float().to(device), batch.edge_index.to(device))
        x_hat = outputs
        loss = criterion(x_hat.float(), batch.x.float().to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    
    train_loss /= len(train_loader.dataset)
    
    gae.eval()  # set the model to evaluation mode
    val_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            x_hat = gae(batch.x.float().to(device), batch.edge_index.to(device))
            loss = criterion(x_hat.float(), batch.x.float().to(device))
            val_loss += loss.item() * batch.num_graphs
    
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gae.state_dict(), './data/gae_model.pt')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
