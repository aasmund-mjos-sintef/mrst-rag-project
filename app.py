import streamlit as st
from graph import *
import io
import streamlit.components.v1 as components

st.set_page_config(layout = "wide", page_title = "MRST Assistant", page_icon = "mrst_logo.png")

mrst_logo, _, title, _, ai_logo = st.columns([2, 3, 4, 3, 1])

with mrst_logo:
    st.image("mrst_logo.png")

with title:
    st.title("MRST Virtual assistant")

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
    chapter_info = state.get('chapter_info')
    images = []

    if figures != None:
        print("Found ", len(figures), " chapters!")
        for c_info, fig in zip(chapter_info, figures):

            buf = io.StringIO()
            fig.savefig(buf, format = 'svg')
            image = buf.getvalue()
            images.append((c_info, image))
            buf.close()

    st.session_state.figures = images

st.markdown("#### Hi, I am an assistant made by SINTEF for the Matlab Reservoir Simulation Toolbox. I can assist you by guiding you to which MRST developers you should contact based on your specific problem, and where in the MRST textbooks you might be able to get help regarding your problem. Please write down any problems you might have and press enter to run!")
query = st.text_input("", "", key = "query", on_change=run_graph)
response_area = st.markdown("")
response_area.markdown(st.session_state.response)

for c_info, img in st.session_state.figures:
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
    
    st.markdown(f"Map over chapter {c_info[0]} in {c_info[1]}. A green node means that I found relevant content in that chapter. You can zoom in by scrolling.")