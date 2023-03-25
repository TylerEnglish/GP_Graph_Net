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
    st.write("# GNNs for Drug Discovery: Analyzing Zinc Database")
    st.write("Today, we'll be discussing how a Graph Neural Network, or GNN, can be used to analyze the Zinc database and identify potential drug candidates. The Zinc database is a large collection of purchasable compounds that can be used for drug discovery. However, identifying potential drug candidates from this database is a challenging task, as the compounds are represented as nodes in a graph with various relationships, such as similarity or chemical properties, represented as edges in the graph.")
    st.write("## What we are trying to solve")
    st.write("Quantum chemistry is a field of study that seeks to explain and predict the behavior of molecules and atoms using quantum mechanics. This app will help you learn about the basics of quantum mechanics and how it applies to chemistry.")
    st.write("## Application in the real world")
    st.write("Quantum chemistry has many applications in the real world. For example, it can be used to predict the behavior of chemical reactions, design new drugs, and develop new materials with specific properties.")


welcome_page()
