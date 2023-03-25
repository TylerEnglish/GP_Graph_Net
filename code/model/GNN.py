import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myGCNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(myGCNConv, self).__init__(aggr='add')
        # self.lin = torch.nn.Linear(in_channels, out_channels)
        # self.add_module('lin', self.lin)
        self.add_module('conv', GCNConv(in_channels, hidden_channels))
        self.add_module('lin2', torch.nn.Linear(hidden_channels, hidden_channels))
        self.add_module('relu', torch.nn.ReLU())
        self.add_module('conv2', GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        # x = self.lin(x)
        x = self.conv(x, edge_index) # GCNConv
        x = self.relu(x) # ReLU
        x = self.conv2(x, edge_index) # GCNConv

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
print(data)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)

# Further split the training set into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=123)

batch_size = 32

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Define the model
gnn = myGCNConv(data.num_features, 16, data.num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = gnn.to(device)

# Define the loss function
# criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

# Set the model to training mode
gnn.train()

# Define the number of training epochs
num_epochs = 100

gnn = gnn.to(device)

train_losses, val_losses = [], []

# Train loop with validation
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
best_val_loss = float('inf')

for epoch in range(num_epochs):
    gnn.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
        loss = criterion(x_hat, batch.y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
        # print("=", end="")
    
    train_loss /= len(train_data)
    print("")
    
    gnn.eval()
    val_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
            loss = criterion(x_hat, batch.y.to(device))
            val_loss += loss.item() * batch.num_graphs
            # print(">", end="")
    
    val_loss /= len(val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gnn.state_dict(), './data/gcn_model.pt')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

gnn.load_state_dict(torch.load('./data/gcn_model.pt'))
gnn.eval()
test_loss = 0
for batch in test_loader:
    with torch.no_grad():
        x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
        loss = criterion(x_hat, batch.y.to(device))
        test_loss += loss.item() * batch.num_graphs
        
test_loss /= len(test_data)

print(f"Test Loss: {test_loss:.4f}")

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.show()
plt.savefig('./data/gcnloss.png')

data = data[0]

data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)

print("x: \n", data.x)
print("edge_index: \n", data.edge_index)
print("Output: ")
print(gnn.forward(data.x, data.edge_index))