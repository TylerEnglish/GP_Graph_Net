import streamlit as st
import streamlit.components.v1 as components
import requests
from streamlit_lottie import st_lottie
import os


# Page 2: What is a Graph Neural Network


def graph_neural_network():
    st.write("# What is a Graph Neural Network")
    st.write("A graph neural network (GNN) is a type of neural network designed to work with data represented in the form of a graph. In a graph, data is represented as a set of interconnected nodes, where each node represents an entity and edges represent relationships between entities.")
    st.write("GNNs operate by propagating information through the graph structure using a set of learnable functions. Each node in the graph is associated with a feature vector, which represents the characteristics of the corresponding entity. The GNN processes the node features and the edge connections to compute a new set of node features that incorporate information from the neighboring nodes in the graph.")
    st.write("The key idea behind GNNs is to leverage the graph structure to improve the representation of each node by considering the local and global connectivity patterns of the graph. This allows GNNs to capture complex relationships between entities and learn hierarchical representations that can be used for various downstream tasks, such as node classification, graph classification, and link prediction.")

    st.write("## What is Graph Convolutional Network")
    st.write("GCN stands for Graph Convolutional Network, which is a type of neural network that operates on graphs. A graph is a mathematical structure that represents a set of objects (nodes) and their connections (edges). In a GCN, each node in a graph is represented by a feature vector, which describes some characteristics of that node. The GCN uses these feature vectors to learn a function that maps each node's features to a new set of features that represent the node's relationship with its neighbors.")
    st.write("The basic operation of a GCN is a graph convolution, which is similar to the convolution operation used in image processing. In a graph convolution, each node aggregates information from its neighbors by taking a weighted sum of their feature vectors, where the weights are learned during training. The node then applies a non-linear function (such as a ReLU activation) to the aggregated information to produce its new feature vector.")
    st.write("The process of graph convolution is repeated multiple times, with each iteration allowing nodes to aggregate information from farther away in the graph. The final output of the GCN is a set of feature vectors for all the nodes in the graph, which can be used for various downstream tasks such as node classification, link prediction, or graph-level classification")
    st.write("Overall, a GCN provides a way to learn representations of nodes in a graph that take into account the graph's structure and the relationships between nodes. This makes GCNs useful for a wide range of applications in fields such as computer vision, natural language processing, and social network analysis.")

    st.write("## What is a Graph Auto Encoder")
    st.write("GAE stands for Graph Autoencoder, which is a type of neural network used for unsupervised representation learning on graphs. The goal of GAE is to learn a low-dimensional representation of a graph that preserves important structural information while discarding irrelevant noise.")
    st.write("GAE is a two-stage process: encoder and decoder. In the encoder stage, GAE first maps each node in the graph to a low-dimensional vector representation using a neural network. These node embeddings are learned by optimizing a reconstruction loss that measures the difference between the original adjacency matrix of the graph and its reconstructed version, which is computed using the decoder stage.")
    st.write("In the decoder stage, GAE reconstructs the adjacency matrix of the graph from the learned node embeddings. This is achieved using a different neural network that takes as input the node embeddings and outputs a reconstructed adjacency matrix.")
    st.write("By training the GAE to reconstruct the adjacency matrix, the model learns to encode the graph's structural information into the low-dimensional embeddings. The node embeddings can then be used for various downstream tasks, such as node classification or link prediction.")


graph_neural_network()
# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


# lottie_coding = load_lottieurl(
#     "https://assets10.lottiefiles.com/packages/lf20_itilDAyVNt.json")
