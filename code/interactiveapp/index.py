import streamlit as st
import streamlit.components.v1 as components
import os
from visual_displays.welcome import welcome_page
from visual_displays.gnn import graph_neural_network
from visual_displays.code import code_and_validation
from visual_displays.demo import demonstration

# Sidebar Section
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

# import streamlit as st
# import streamlit.components.v1 as components
# import os
# from visual_displays.demo import welcome_page
# from visual_displays.gnn import graph_neural_network
# from visual_displays.code import code_and_validation
# from visual_displays.demo import demonstration

# # Sidebar Section
# st.sidebar.header("Quantum Chemistry")

# # Define CSS styles for the buttons
# button_styles = {
#     "background-color": "#FFFFFF",
#     "border": "1px solid #CCCCCC",
#     "border-radius": "5px",
#     "padding": "10px",
#     "font-weight": "bold",
#     "font-size": "18px",
#     "margin-bottom": "10px",
# }

# # Define CSS styles for the active button
# active_button_styles = {
#     "background-color": "#5C6BC0",
#     "color": "#FFFFFF",
# }

# # Define a function to highlight the active button


# def highlight_active_button(button_name):
#     if button_name in st.session_state.active_button:
#         return active_button_styles
#     else:
#         return button_styles


# # Initialize the active button to the first button
# if "active_button" not in st.session_state:
#     st.session_state.active_button = "Welcome Page"

# # Add the buttons to the sidebar
# if st.sidebar.button("Welcome Page", key="Welcome Page", style=highlight_active_button("Welcome Page")):
#     st.session_state.active_button = "Welcome Page"
# if st.sidebar.button("What is a Graph Neural Network", key="What is a Graph Neural Network", style=highlight_active_button("What is a Graph Neural Network")):
#     st.session_state.active_button = "What is a Graph Neural Network"
# if st.sidebar.button("Code and Validation", key="Code and Validation", style=highlight_active_button("Code and Validation")):
#     st.session_state.active_button = "Code and Validation"
# if st.sidebar.button("Demonstration", key="Demonstration", style=highlight_active_button("Demonstration")):
#     st.session_state.active_button = "Demonstration"

# # Display the selected page
# if st.session_state.active_button == "Welcome Page":
#     welcome_page()
# elif st.session_state.active_button == "What is a Graph Neural Network":
#     graph_neural_network()
# elif st.session_state.active_button == "Code and Validation":
#     code_and_validation()
# elif st.session_state.active_button == "Demonstration":
#     demonstration()
