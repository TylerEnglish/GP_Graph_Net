import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx



class myEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(myEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class myDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(myDecoder, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class myGAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(myGAE, self).__init__()
        self.encoder = myEncoder(input_dim, hidden_dim)
        self.decoder = myDecoder(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x)
        return x

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './data/gae_model.pt'

    data = pickle.load(open('./data/dataset.pkl', 'rb'))
    print(data)

    data = data[:100000]
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
    num_epochs = 100

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

    # Plot the training and validation losses
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'./data/pics/gae_loss.png')





    gae = myGAE(data.num_features, 16, data.num_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gae = gae.to(device)

    # Load the saved model
    gae.load_state_dict(torch.load('./data/gae_model.pt'))


    # Convert a PyTorch Geometric graph to a NetworkX graph
    G = to_networkx(batch, to_undirected=True)

    # Plot the input graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=50)
    plt.title('Input Graph')
    plt.savefig(f'./data/pics/input_chart.png')

    # Evaluate the trained model on the test set
    gae.eval()
    outputs = []
    for batch in test_loader:
        with torch.no_grad():
            output = gae(batch.x.float().to(device), batch.edge_index.to(device))
            output = output.cpu().numpy().tolist()
            outputs += output

    # Convert the output to a list of edges
    edges = []
    for output in outputs:
        edges.append([(i, j) for i, j in enumerate(output) if j > 0.5])

    # Create a new NetworkX graph object
    output_graph = nx.Graph()

    # Add nodes to the graph
    output_graph.add_nodes_from(range(data.num_features))

    # Add edges to the graph
    for e in edges:
        output_graph.add_edges_from(e)

    # Plot the output graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(output_graph, seed=42)
    nx.draw(output_graph, pos, node_size=50)
    plt.title('Output Graph')
    plt.savefig(f'./data/pics/output_chart.png')

def demo(n):
    data = pickle.load(open('./data/dataset.pkl', 'rb'))

    # Select a single input
    data = data[n]

    gae = myGAE(data.num_features, 16, data.num_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gae = gae.to(device)
    # Load the saved model
    gae.load_state_dict(torch.load('./data/gae_model.pt'))

    # Convert a PyTorch Geometric graph to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Plot the input graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=50)
    plt.title('Input Graph')
    plt.savefig(f'./data/pics/demo_input_chart.png')

    # Evaluate the trained model on the input
    gae.eval()
    with torch.no_grad():
        output = gae(data.x.float().to(device), data.edge_index.to(device))
        output = output.cpu().numpy().tolist()

    # Convert the output to a list of edges
    edges = [(i, j) for i, j in enumerate(output[0]) if j > 0.5]

    # Create a new NetworkX graph object
    output_graph = nx.Graph()

    # Add nodes to the graph
    output_graph.add_nodes_from(range(data.num_features))

    # Add edges to the graph
    output_graph.add_edges_from(edges)

    # Plot the output graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(output_graph, seed=42)
    nx.draw(output_graph, pos, node_size=50)
    plt.title('Output Graph')
    plt.savefig(f'./data/pics/demo_output_chart.png')


if __name__ == '__main__':
    demo(2018)