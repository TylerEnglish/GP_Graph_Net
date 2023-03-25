import streamlit as st
import streamlit.components.v1 as components
import os

#Sidebar Section
st.sidebar.header("Quantum Chemistry")

button1 = st.sidebar.button("Welcome Page")
button2 = st.sidebar.button("What is a Graph Neural Network")
button3 = st.sidebar.button("Code and Validation")
button4 = st.sidebar.button("Demonstration")

if button1:
    welcome_page()
if button2:
    graph_neural_network()
if button3:
    code_and_validation()
if button4:
    demonstration()