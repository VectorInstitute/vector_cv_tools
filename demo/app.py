import tempfile

import streamlit as st
import session_state

from demo_registry import get_demo, get_all_demos


def main():
    st.set_page_config(page_title="Project Alium",
                       page_icon="assets/vertical_logo.jpg",
                       layout='wide',
                       initial_sidebar_state='expanded')

    state = session_state.get_state()
    st.sidebar.header('Project Alium')
    selected_demo = st.sidebar.selectbox('Select a Demo', get_all_demos())
    demo_func = get_demo(selected_demo)
    demo_func(state)

    state.sync()


def hide_streamlit_widgets():
    """
    hides widgets that are displayed by streamlit when running
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
