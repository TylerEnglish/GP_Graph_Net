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

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

        # Step 5: Apply global max pooling
        x = self.aggregate(x, torch.zeros(x.size(0), dtype=torch.long, device=device))

        # Step 6: Apply classifier
        # x = self.classifier(x)
        return x


    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


data = pickle.load(open('./data/dataset.pkl', 'rb'))

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

gnn = GCNConv(data.num_features, data.num_classes)
gnn = gnn.to(device)

optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

# Train the model
train_loss_history = []
val_loss_history = []


num_epochs = 10

for epoch in range(num_epochs):
    # Training
    train_loss = 0
    gnn.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = gnn(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)

    # Validation
    val_loss = 0
    gnn.eval()

    for data in val_loader:
        data = data.to(device)
        with torch.no_grad():
            out = gnn(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')




plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the chart
plt.savefig('loss_chart.png')