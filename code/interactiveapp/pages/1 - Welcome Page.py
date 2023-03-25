import streamlit as st
import streamlit.components.v1 as components
import os

# Page 1: Welcome Page


def welcome_page():
    st.write("# Quantum Chemistry")
    st.write("Welcome to the Quantum Chemistry app. This app is designed to help you understand the basics of quantum chemistry and its applications in the real world.")
    st.write("## What we are trying to solve")
    st.write("Quantum chemistry is a field of study that seeks to explain and predict the behavior of molecules and atoms using quantum mechanics. This app will help you learn about the basics of quantum mechanics and how it applies to chemistry.")
    st.write("## Application in the real world")
    st.write("Quantum chemistry has many applications in the real world. For example, it can be used to predict the behavior of chemical reactions, design new drugs, and develop new materials with specific properties.")

welcome_page()