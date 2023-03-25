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
# Load the dataset from file using pickle
try:
    with open('./data/dataset.pkl', 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print("Error: dataset file not found.")
    exit(1)
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)



x = torch.tensor(data.x, dtype=torch.float)
edge_index = torch.tensor(data.edge_index, dtype=torch.long)
y = torch.tensor(data.y, dtype=torch.float)

train_mask, test_mask = train_test_split(
    range(data.num_nodes), test_size=0.2, random_state=123)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_mask] = True
data.test_mask[test_mask] = True

model = GAE(
    Encoder(in_channels=1, hidden_channels=16, out_channels=1).to(device),
    Decoder(in_channels=1, out_channels=1).to(device)
).to(device)

# Train loop with validation
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses, val_losses = [], []

for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[data.train_mask], y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        train_loss = criterion(out[data.train_mask], y[data.train_mask])
        test_loss = criterion(out[data.test_mask], y[data.test_mask])
        train_losses.append(train_loss.item())
        val_losses.append(test_loss.item())
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Print out test input, Wanna explore test_loader data
# i = 0
# graphs = []
# single_input = []
# for batch in test_loader:
#     if i < 10:
#         networkx_graph = to_networkx(batch)
#         print(type(networkx_graph))
#         graphs.append(networkx_graph)

#         single_input.append(batch)
#         print(batch)
#     i+=1

# # display input
# import os
# if not os.path.exists('./data/pics'):
#     os.makedirs('./data/pics')

# plt.figure(figsize=(15, 5))
# for i in range(len(single_input)):
#     plt.subplot(2, 5, i+1)
#     nx.draw(graphs[i])

# plt.savefig(f'./data/pics/input_charts.png')

# data =  Data(x=single_input[0].x, edge_index=single_input[0].edge_index)
# print("\nSample data point:")
# print("Number of nodes:", data.num_nodes)
# print("Node features shape:", data.x.shape)
# print("Edge features shape:", data.edge_index.shape)
# print("Edges:", data.edge_index)
# print(data)


# # feed in each data point through model and output it
# # gnn.eval()
# # i = 0
# # o_graphs = []
# # single_output = []
# # for batch in test_loader:
# #     if i < 10:
# #         with torch.no_grad():
# #             x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
# #             # output the predicted values
# #             print("Input:", batch.x)
# #             print("Output:", x_hat)


# # Feed in each data point through model and output it
# gnn.eval()
# i = 0
# o_graphs = []
# single_output = []

# for batch in test_loader:
#     if i < 10:
#         with torch.no_grad():
#             x_hat = gnn(batch.x.to(device), batch.edge_index.to(device))
#             # Output the predicted values
#             print("Input:", batch.x)
#             print("Output:", x_hat)
            
#             # Store input and output values
#             single_output.append(x_hat.cpu())
            
#             # Convert output tensor to networkx graph
#             output_graph = to_networkx(Data(x=x_hat.cpu(), edge_index=batch.edge_index.cpu()))
#             o_graphs.append(output_graph)
#     i += 1


# for i in range(len(single_output)):
#     plt.subplot(2, 5, i+1)
#     nx.draw(o_graphs[i])

# plt.savefig(f'./data/pics/_charts.png')