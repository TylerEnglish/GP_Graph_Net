import streamlit as st
import streamlit.components.v1 as components
import os
# from visual_displays.welcome import welcome_page
# from visual_displays.gnn import graph_neural_network
# from visual_displays.code import code_and_validation
# from visual_displays.demo import demonstration

# Sidebar Section
st.sidebar.header("Zinc Molecular Weight")

def welcome_page():
    st.write("# GNN Zinc Molecular Weight Prediction")
    st.write("Welcome to the Quantum Chemistry app. This app is designed to help you understand the basics of quantum chemistry and its applications in the real world.")
    st.write("## What we are trying to solve")
    st.write("Quantum chemistry is a field of study that seeks to explain and predict the behavior of molecules and atoms using quantum mechanics. This app will help you learn about the basics of quantum mechanics and how it applies to chemistry.")
    st.write("## Application in the real world")
    st.write("Quantum chemistry has many applications in the real world. For example, it can be used to predict the behavior of chemical reactions, design new drugs, and develop new materials with specific properties.")

welcome_page()




# button1 = st.sidebar.button("Welcome Page")
# button2 = st.sidebar.button("What is a Graph Neural Network")
# button3 = st.sidebar.button("Code and Validation")
# button4 = st.sidebar.button("Demonstration")

# if button1:
#     welcome_page()
# if button2:
#     graph_neural_network()
# if button3:
#     code_and_validation()
# if button4:
#     demonstration()

# with st.sidebar:
#     add_radio = st.radio("Select a page", ("Welcome Page",
#                          "What is a Graph Neural Network", "Code and Validation", "Demonstration"))

# if add_radio == "What is a Graph Neural Network":
#     graph_neural_network()
# elif add_radio == "Code and Validation":
#     code_and_validation()
# elif add_radio == "Demonstration":
#     demonstration()
# else:
#     welcome_page()
