import streamlit as st
from graph import *
import io
from PIL import Image
from streamlit_image_zoom import image_zoom
import streamlit.components.v1 as components

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
    st.session_state.figures = []

def run_graph():
    query = st.session_state.query
    response_area.text("...")

    state = graph.invoke(State(query = query))

    st.session_state.response = state.get('response')

    figures = state.get('figures')
    images = []

    if figures != None:
        print("Found ", len(figures), " chapters!")
        for fig in figures:

            buf = io.StringIO()
            fig.savefig(buf, format = 'svg')
            image = buf.getvalue()
            images.append(image)
            buf.close()

    st.session_state.figures = images

information_area = st.markdown("## What can I help you with?")
query = st.text_input("", "", key = "query", on_change=run_graph)
response_area = st.text("")
response_area.text(st.session_state.response)

for img in st.session_state.figures:
    components.html("""
<div style="
    border: 2px solid #ccc;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 20px;
    background-color: #ffffff;
    display: inline-block;
    box-sizing: border-box;
    min-height: 100px;
">
    <div id="svg-container" style="width: auto; height: auto; overflow: auto;">
        """ + img + """
    </div>
    <script src="https://unpkg.com/@panzoom/panzoom/dist/panzoom.min.js"></script>
    <script>
        var elem = document.getElementById('svg-container');
        var panzoom = Panzoom(elem, { maxScale: 5, minScale: 1});
        elem.addEventListener('wheel', panzoom.zoomWithWheel);
    </script>
</div>
""", height = 550)
    
    st.markdown("Some Chapter")