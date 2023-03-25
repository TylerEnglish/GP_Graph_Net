import streamlit as st
import streamlit.components.v1 as components
import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

# Page 3: Code and Validation


def code_and_validation():
    st.write("# Code and Validation")
    st.write("In this section, we will discuss the code used in this app and how we validate the results.")
    st.subheader("How we loaded data")
    st.code('''
import pickle
from torch_geometric.data import DataLoader
import torch
from torch_geometric.datasets import ZINC

# Load the dataset
dataset = ZINC(root='data/ZINC')

# Save the dataset as a file
with open('./data/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
    ''', 'python')

    st.subheader('How to load the data in new file')
    try:
        with open('./data/dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("Error: dataset file not found.")
        exit(1)
    except Exception as e:
        print("Error loading dataset:", e)
        exit(1)
    st.code('''
try:
    with open('./data/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print("Error: dataset file not found.")
    exit(1)
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)
    ''', 'python')
    st.subheader('Load in a graph')
    fig = plt.figure(figsize=(20, 15))
    st.code('''
fig = plt.figure(figsize=(20, 15))
data = dataset[n]
networkx_graph = to_networkx(data)
nx.draw_networkx(networkx_graph)
    ''', 'python')

    # Use a slider to select a graph from the dataset
    number = st.slider('Pick a number:', 0, len(dataset)-1)
    data = dataset[number]
    networkx_graph = to_networkx(data)
    nx.draw_networkx(networkx_graph)
    plt.show()
    st.pyplot(fig)
    st.write("## Code and what it does")
    st.write("The code used in this app is written in Python and uses several libraries, including PyTorch, RDKit, and NumPy. The code is used to calculate the dipole moment of a given chemical formula.")
    st.write("## Validating results")
    st.write("To validate the results, we compared the dipole moment calculated by the app with experimental data from the literature. The results were found to be in good agreement.")


code_and_validation()
