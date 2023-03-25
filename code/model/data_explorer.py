import pickle
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

# Load the dataset from file using pickle
try:
    with open('./data/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print("Error: dataset file not found.")
    exit(1)
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)

# Print basic information about the dataset
print("Number of samples:", len(dataset))
print("Number of features:", dataset.num_features)
print(dataset[0])
print("Number of classes:", dataset.num_classes)

# Sample a data point and print information about it
data = dataset[0]
print("\nSample data point:")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Node features shape:", data.x.shape)
print("Edge features shape:", data.edge_attr.shape)
print("Target value:", data.y)

# Demonstrate how to access specific nodes and edges in the graph
print(data.x[:1].shape) # Access the first node's features
print("Shape of sample nodes:", data.x[:5].shape) # Access the features of the first five nodes

# Iterate over the edges and print their source and destination nodes
edge_index = dataset[0].edge_index
for i in range(edge_index.shape[1]):
    src = edge_index[0, i]  # index of the source node
    dst = edge_index[1, i]  # index of the destination node
    print("Edge {}: source={}, destination={}".format(i, src, dst))

# Convert the data to a NetworkX graph for visualization
print(type(data))
networkx_graph = to_networkx(data)
print(type(networkx_graph))

# Visualize multiple graphs
n = 10
fig = plt.figure(figsize=(20, 15))
for i in range(n):
    plt.subplot(2, 5, i+1)
    data = dataset[i]
    networkx_graph = to_networkx(data)
    nx.draw_networkx(networkx_graph)

plt.savefig(f'./data/pics/data_charts.png')



# Read a CSV file using Pandas
try:
    PATH = './data/ZINC/raw/zinc_standard_agent/properties.csv'
    df = pd.read_csv(PATH)
    print(df.info())
except FileNotFoundError:
    print("Error: file not found.")
    exit(1)
except Exception as e:
    print("Error reading file:", e)
    exit(1)
