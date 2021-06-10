import streamlit as st
import markdowns

sys_diag_img = open("./assets/sys_diag.png", "rb").read()


def system_level_diagram(state):
    st.title("System Level Diagram")

    st.markdown(markdowns.sys_intro)

    _, col2, _ = st.beta_columns([1, 4, 1])
    col2.image(sys_diag_img)

    st.markdown(markdowns.acknowledgements)
