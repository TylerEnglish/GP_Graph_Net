import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

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
print(data)
gnn = GCNConv(data.num_features, data.num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = gnn.to(device)

data = data[0]

data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)

print("x: \n", data.x)
print("edge_index: \n", data.edge_index)
print("Output: ")
print(gnn.forward(data.x, data.edge_index))