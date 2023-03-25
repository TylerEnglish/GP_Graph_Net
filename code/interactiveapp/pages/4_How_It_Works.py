import streamlit as st

st.write("""
# How Our Project Works
Our model consists of two parts:
1. The Graph AutoEncoder (GAE)
2. The Graph Convolutional Network (GCN)

### Graph AutoEncoder
The autoencoder takes in an partial graph containing just the basic information about a compound, what atoms are in it and how they're bonded to each other. Then it outputs a complete graph.

So you put a graph in, and get a graph out. What's exciting about that? Well, the complete graph includes all the data described by the compound graphs in the Zinc database, *including* things like chemical relationships and biological effects.

So our output from the GAE tells us predicted drug behaviors of these compounds.

### Graph Convolutional Network
Our convolutional network takes the output from our GAE
""")