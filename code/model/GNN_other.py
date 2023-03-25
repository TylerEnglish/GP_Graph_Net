import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myGCNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(myGCNConv, self).__init__(aggr='add')
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

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
# get subset of data
data = data[:10000]
print(len(data))

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

print("Number of training examples:", len(train_data))
print(train_data)
print("Number of validation examples:", len(val_data))
print("Number of test examples:", len(test_data))

batch_size = 1

# Create data loaders
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=batch_size)
# test_loader = DataLoader(test_data, batch_size=batch_size)

# Define the model
gnn = myGCNConv(data.num_features, 16, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = gnn.to(device)

# Define the loss function
# criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer
# optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

# Set the model to training mode
gnn.train()

# Define the number of training epochs
num_epochs = 100

gnn = gnn.to(device)

train_losses, val_losses = [], []

# Train loop with validation
optimizer = torch.optim.Adam(gnn.parameters(), lr=.5)

# optimizer = optim.SGD(gnn.parameters(), lr=.5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# criterion = torch.nn.MSELoss()
criterion = torch.nn.MSELoss(reduction='mean')
# criterion = torch.nn.MSELoss(size_average=batch_size, reduce=True, reduction="mean")
best_val_loss = float('inf')

for epoch in range(num_epochs):
    gnn.train()
    train_loss = 0
    for graph in train_data:
        # print(batch)
        optimizer.zero_grad()
        x_hat = gnn(graph.x.float().to(device), graph.edge_index.to(device))
        # print("x_hat shape:", x_hat.shape)
        # print("batch.y shape:", batch.y.float().to(device).shape)
        

        loss = criterion(x_hat[0], graph.y.to(device))

        loss.backward()
        optimizer.step()
        train_loss += loss.item() #* batch.num_graphs
        # print("=", end="")
    
    train_loss /= len(train_data)
    print("")
    
    gnn.eval()
    val_loss = 0
    for graph in val_data:
        with torch.no_grad():
            x_hat = gnn(graph.x.float().to(device), graph.edge_index.to(device))
            loss = criterion(x_hat[0], graph.y.to(device))

            val_loss += loss.item() #* batch.num_graphs
            # print(">", end="")
    
    val_loss /= len(val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gnn.state_dict(), './data/gcn_model.pt')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    scheduler.step()

gnn.load_state_dict(torch.load('./data/gcn_model.pt'))
gnn.eval()
test_loss = 0
for graph in test_data:
    with torch.no_grad():
        x_hat = gnn(graph.x.float().to(device), graph.edge_index.to(device))
        loss = criterion(x_hat[0], graph.y.to(device))

        test_loss += loss.item() #* batch.num_graphs
        
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

data.x = data.x.float().to(device)
data.edge_index = data.edge_index.long().to(device)

print("x: \n", data.x)
print("edge_index: \n", data.edge_index)
print("Output: ")
print(gnn.forward(data.x, data.edge_index))