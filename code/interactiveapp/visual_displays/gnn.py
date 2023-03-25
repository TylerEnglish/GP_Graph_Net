import streamlit as st
import streamlit.components.v1 as components
import os

# Page 2: What is a Graph Neural Network

def graph_neural_network():
    st.write("# What is a Graph Neural Network")
    st.write("Graph neural networks (GNNs) are a type of neural network that can be used to analyze and make predictions about graph-structured data.")
    st.write("## What is a graph neural network")
    st.write("A graph neural network is a type of neural network that operates on graph-structured data. Graphs are a type of data structure that consist of nodes (also known as vertices) and edges (also known as links).")
    st.write("## How does it work")
    st.write("A graph neural network works by passing messages between the nodes of a graph. The messages contain information about the features of the nodes and their connections to other nodes. The network uses this information to make predictions about the graph.")
