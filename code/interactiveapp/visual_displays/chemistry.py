import streamlit as st
import streamlit.components.v1 as components
import os

_RELEASE = True

# Create dropdown menu
# pages = {"Welcome Page": welcome_page,
#          "What is a Graph Neural Network": graph_neural_network,
#          "Code and Validation": code_and_validation,
#          "Demonstration": demonstration}
# page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# pages[page]()


# Making the side buttons instead of dropdown
# welcome_page()





# Display the selected page
if not _RELEASE:
    _component_func = components.declare_component(
        "option_menu",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/dist")
    _component_func = components.declare_component(
        "option_menu", path=build_dir)


def option_menu(menu_title, options, default_index=0, menu_icon=None, icons=None, orientation="vertical",
                styles=None, key=None):
    component_value = _component_func(options=options,
                key=key, defaultIndex=default_index, icons=icons, menuTitle=menu_title,
                menuIcon=menu_icon, default=options[default_index],
                orientation=orientation, styles=styles)
    return component_value

st.set_page_config(page_title="WWPapers ChatBot", page_icon=":robot_face:", layout='centered')


selected = option_menu(
    menu_title=None,
    options=["Home", "How It Works", "Stats", "Take Aways"],
    icons=["house", "cpu", "bar-chart"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={

    }
)

if selected == "Home":
    home_page()
if selected == "How It Works":
    how_it_works_page()
if selected == "Stats":
    stats_page()
if selected == "Take Aways":
    take_aways_page()
