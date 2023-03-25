import pickle

# Load the dataset from file
with open('./data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Data Exploration
print("Number of samples:", len(dataset))
print("Number of features:", dataset.num_features)
print("Number of classes:", dataset.num_classes)

# Sample a data point
data = dataset[0]
print("\nSample data point:")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Node features shape:", data.x.shape)
print("Edge features shape:", data.edge_attr.shape)
print("Target value:", data.y)

# Get the edge index for the first data point
edge_index = dataset[0].edge_index

# Iterate over the edges
for i in range(edge_index.shape[1]):
    src = edge_index[0, i]  # index of the source node
    dst = edge_index[1, i]  # index of the destination node
    print("Edge {}: source={}, destination={}".format(i, src, dst))