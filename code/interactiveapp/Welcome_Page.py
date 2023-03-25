import streamlit as st
import streamlit.components.v1 as components
import os
# from visual_displays.welcome import welcome_page
# from visual_displays.gnn import graph_neural_network
# from visual_displays.code import code_and_validation
# from visual_displays.demo import demonstration

# Sidebar Section
# st.sidebar.title("Zinc Molecular Weight")


def welcome_page():
    st.write("# Graph Neural Networks for Drug Discovery")
    st.write("Today, we'll be discussing how a Graph Neural Network, or GNN, can be used to analyze the Zinc database and identify potential drug candidates. The Zinc database is a large collection of purchasable compounds that can be used for drug discovery. However, identifying potential drug candidates from this database is a challenging task, as the compounds are represented as nodes in a graph with various relationships, such as similarity or chemical properties, represented as edges in the graph.")
    st.write("## What we are trying to solve")
    st.write("Our objective is to use a Graph Neural Network (GNN) to analyze the Zinc database and identify potential drug candidates. The Zinc database is a large collection of purchasable compounds that can be used for drug discovery, but traditional methods of identifying potential drug candidates from this database are limited. By leveraging the graph structure of the database and using a GNN, we aim to overcome these limitations and discover new drugs that could save lives.")


welcome_page()
