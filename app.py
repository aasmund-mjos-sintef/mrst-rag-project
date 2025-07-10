import streamlit as st
from graph import *
import io

st.set_page_config(layout = "wide", page_title = "MRST Assistant", page_icon = "mrst_logo.png")

mrst_logo, _, title, _, ai_logo = st.columns([2, 3, 4, 3, 1])

with mrst_logo:
    st.image("mrst_logo.png")

with title:
    st.title("Virtual assistant")

with ai_logo:
    st.image("ai_logo.png")

if "response" not in st.session_state:
    st.session_state.response = ""

if "figures" not in st.session_state:
    st.session_state.figures = [""]

def run_graph():
    query = st.session_state.query
    response_area.text("...")

    state = graph.invoke(State(query = query))

    st.session_state.response = state.get('response')

    figures = state.get('figures')
    svg_figures = []

    if figures != None:
        for fig in figures:
            buf = io.StringIO()
            fig.savefig(buf, format = "svg")
            svg_figures.append(buf.getvalue())
            buf.close()

    st.session_state.figures = svg_figures


query = st.text_input("What can I help you with? ", "", key = "query", on_change=run_graph)
response_area = st.text("")
response_figures = st.markdown("")
response_area.text(st.session_state.response)

svg_content = ""
for svg_fig in st.session_state.figures:
    svg_content += svg_fig

response_figures.markdown(svg_content, unsafe_allow_html=True)