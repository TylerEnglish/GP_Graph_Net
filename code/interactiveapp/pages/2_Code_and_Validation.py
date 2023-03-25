import streamlit as st
import streamlit.components.v1 as components
import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx
from PIL import Image

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
    st.header('GAE')
    st.code('''
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
    ''','python')
    st.subheader('Training')
    st.code('''
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
    ''','python')
    st.header("GCN")
    st.code('''

    ''', 'python')

    st.subheader('Training')
    st.code('''

    ''','python')


code_and_validation()
