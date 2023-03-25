import streamlit as st
import streamlit.components.v1 as components
import os

_RELEASE = True
# Page 1: Welcome Page


def welcome_page():
    st.write("# Quantum Chemistry")
    st.write("Welcome to the Quantum Chemistry app. This app is designed to help you understand the basics of quantum chemistry and its applications in the real world.")
    st.write("## What we are trying to solve")
    st.write("Quantum chemistry is a field of study that seeks to explain and predict the behavior of molecules and atoms using quantum mechanics. This app will help you learn about the basics of quantum mechanics and how it applies to chemistry.")
    st.write("## Application in the real world")
    st.write("Quantum chemistry has many applications in the real world. For example, it can be used to predict the behavior of chemical reactions, design new drugs, and develop new materials with specific properties.")

# Page 2: What is a Graph Neural Network


def graph_neural_network():
    st.write("# What is a Graph Neural Network")
    st.write("Graph neural networks (GNNs) are a type of neural network that can be used to analyze and make predictions about graph-structured data.")
    st.write("## What is a graph neural network")
    st.write("A graph neural network is a type of neural network that operates on graph-structured data. Graphs are a type of data structure that consist of nodes (also known as vertices) and edges (also known as links).")
    st.write("## How does it work")
    st.write("A graph neural network works by passing messages between the nodes of a graph. The messages contain information about the features of the nodes and their connections to other nodes. The network uses this information to make predictions about the graph.")

# Page 3: Code and Validation


def code_and_validation():
    st.write("# Code and Validation")
    st.write("In this section, we will discuss the code used in this app and how we validate the results.")
    st.write("## Code and what it does")
    st.write("The code used in this app is written in Python and uses several libraries, including PyTorch, RDKit, and NumPy. The code is used to calculate the dipole moment of a given chemical formula.")
    st.write("## Validating results")
    st.write("To validate the results, we compared the dipole moment calculated by the app with experimental data from the literature. The results were found to be in good agreement.")

# Page 4: Demonstration


def demonstration():
    st.write("# Demonstration")
    st.write("In this section, you can enter a chemical formula and the app will calculate the dipole moment (MU) for you.")
    # Input box for chemical formula
    chemical_formula = st.text_input("Enter a chemical formula")
    if chemical_formula:
        # Calculate dipole moment and display the result
        dipole_moment = calculate_dipole_moment(chemical_formula)
        st.write(
            f"The dipole moment for {chemical_formula} is {dipole_moment}.")

# Define a function to calculate the dipole moment


def calculate_dipole_moment(chemical_formula):
    # Code to calculate the dipole moment
    return


st.sidebar.header("Quantum Chemistry")
# Create dropdown menu
# pages = {"Welcome Page": welcome_page,
#          "What is a Graph Neural Network": graph_neural_network,
#          "Code and Validation": code_and_validation,
#          "Demonstration": demonstration}
# page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# pages[page]()


# Making the side buttons instead of dropdown
# welcome_page()

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

# # Display the selected page
# if not _RELEASE:
#     _component_func = components.declare_component(
#         "option_menu",
#         url="http://localhost:3001",
#     )
# else:
#     parent_dir = os.path.dirname(os.path.abspath(__file__))
#     build_dir = os.path.join(parent_dir, "frontend/dist")
#     _component_func = components.declare_component(
#         "option_menu", path=build_dir)


# def option_menu(menu_title, options, default_index=0, menu_icon=None, icons=None, orientation="vertical",
#                 styles=None, key=None):
#     component_value = _component_func(options=options,
#                 key=key, defaultIndex=default_index, icons=icons, menuTitle=menu_title,
#                 menuIcon=menu_icon, default=options[default_index],
#                 orientation=orientation, styles=styles)
#     return component_value

# st.set_page_config(page_title="WWPapers ChatBot", page_icon=":robot_face:", layout='centered')


# selected = option_menu(
#     menu_title=None,
#     options=["Home", "How It Works", "Stats", "Take Aways"],
#     icons=["house", "cpu", "bar-chart"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
#     styles={

#     }
# )

# if selected == "Home":
#     home_page()
# if selected == "How It Works":
#     how_it_works_page()
# if selected == "Stats":
#     stats_page()
# if selected == "Take Aways":
#     take_aways_page()
