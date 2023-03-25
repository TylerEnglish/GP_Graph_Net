import streamlit as st
import streamlit.components.v1 as components
import os

# Page 3: Code and Validation


def code_and_validation():
    st.write("# Code and Validation")
    st.write("In this section, we will discuss the code used in this app and how we validate the results.")
    st.write("## Code and what it does")
    st.write("The code used in this app is written in Python and uses several libraries, including PyTorch, RDKit, and NumPy. The code is used to calculate the dipole moment of a given chemical formula.")
    st.write("## Validating results")
    st.write("To validate the results, we compared the dipole moment calculated by the app with experimental data from the literature. The results were found to be in good agreement.")

code_and_validation()