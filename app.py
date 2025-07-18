import streamlit as st
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout = "wide", page_title = "MRST Assistant", page_icon = "mrst_logo.png")

mrst_logo, _, title, _, ai_logo = st.columns([2, 3, 4, 3, 1])

with mrst_logo:
    st.image("mrst_logo.webp")

with title:
    st.title("MRST Virtual assistant")

with ai_logo:
    st.image("langgraph_logo.png")

if "query" not in st.session_state:
    st.session_state.query = ""

if "code_query" not in st.session_state:
    st.session_state.code_query = ""

if "response" not in st.session_state:
    st.session_state.response = ""

if "figures" not in st.session_state:
    st.session_state.figures = []

if "authors" not in st.session_state:
    st.session_state.authors = []

if "github_authors" not in st.session_state:
    st.session_state.github_authors = []

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

if "auto_query" not in st.session_state:
    st.session_state.auto_query = ""

st.markdown("#### Hi, I am an assistant made by SINTEF for the Matlab Reservoir Simulation Toolbox. I can assist you by guiding you to which MRST developers you should contact based on your specific problem, and where in the MRST textbooks you might be able to get help regarding your problem.")

query, code_query, button = st.columns([6,6,1])
with query:
    query = st.text_area(label = "Query",value = st.session_state.query, key = "query", height = 300, label_visibility="hidden", placeholder="Please write down any problems you might have")
with code_query:
    code_query = st.text_area(label = "Code Query",value = st.session_state.code_query, key = "code_query", height = 300, label_visibility="hidden", placeholder="Please write down any matlab code for context")

from graph import *
import io

def run_graph():
    response_area.text("Generating answer...")
    query = st.session_state.query
    code_query = st.session_state.code_query
    auto_query = st.session_state.auto_query
    st.session_state.auto_query = None

    if auto_query == None:
        auto_query = ""

    if query != "" or code_query != "" or auto_query != "":
        if auto_query != "":
            state = graph.invoke(State(query = "Give me a brief summary of what work " + auto_query + " does", code_query= "", query_description = QueryDescriptionWithTools(keywords = [], authors=[auto_query], problem_description="", tools=False, tools_input=""), start_node="RetrieveAuthorNode"))
        else:
            state = graph.invoke(State(query = query, code_query=code_query, start_node="InformationNode"))

        st.session_state.suggestions = state.get('suggestions')
        total_response = ""

        author_response = state.get('author_response')
        if author_response != None:
            total_response += f" \n\n#### Relevant information about {", ".join(state.get('query_description').authors)}  \n\n"
            total_response += author_response

        book_response = state.get('book_response')
        if book_response != None:
            total_response += "\n\n#### Relevant information in the MRST textbooks \n\n"
            total_response += book_response

        authors_relevance_score = state.get('authors_relevance_score')
        if authors_relevance_score != None: 
            if len(authors_relevance_score.keys()):
                authors = sorted(list(zip(authors_relevance_score.keys(), authors_relevance_score.values())), key = lambda x: x[1])
                total_n_authors = len(authors)
                n = min([total_n_authors, 5])
                st.session_state.authors  = authors[total_n_authors-n:][::-1] if total_n_authors else []
        else:
            st.session_state.authors = []

        github_authors_relevance_score = state.get('github_authors_relevance_score')
        if github_authors_relevance_score != None:
            if len(github_authors_relevance_score.keys()):
                authors = sorted(list(zip(github_authors_relevance_score.keys(), github_authors_relevance_score.values())), key = lambda x: x[1])
                total_n_authors = len(authors)
                n = min([total_n_authors, 5])
                st.session_state.github_authors  = authors[total_n_authors-n:][::-1] if total_n_authors else []
        else:
            st.session_state.github_authors = []

        st.session_state.response = total_response

        figures = state.get('figures')
        chapter_info = state.get('chapter_info')
        images = []

        if figures != None:
            for c_info, fig in zip(chapter_info, figures):

                buf = io.StringIO()
                fig.savefig(buf, format = 'svg', facecolor = "#faf9f7")
                image = buf.getvalue()
                images.append((c_info, image))
                buf.close()

        st.session_state.figures = images

def reset_func():
    st.session_state.query = ""
    st.session_state.code_query = ""

with button:
    st.button(label = "Generate", on_click=run_graph, type = "primary")
    st.button(label = "Reset", on_click=reset_func)

response_area = st.markdown(st.session_state.response)

for c_info, img in st.session_state.figures:
    components.html("""
<div style="
    padding: 10px;
    margin-bottom: 20px;
    background-color: #faf9f7;
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

papers, github, _ = st.columns([6,6,1])

with papers:

    if len(st.session_state.authors):
        st.markdown("### Based on the retrieved MRST papers, I would recommend reaching out to: ")

    for a, s in st.session_state.authors:
        st.markdown(f'#### {a}')
        st.markdown('With a relevance score of ' + str(s))

with github:

    if len(st.session_state.github_authors):
        st.markdown("### Based on the relevant commits in the MRST github repository, I would recommend reaching out to: ")

    for a, s in st.session_state.github_authors:
        st.markdown(f'#### {a}')
        st.markdown('With '+ str(s) + ' commits in retrieved folders')

if len(st.session_state.suggestions):
    st.markdown('#### Click on the authors you want to learn more about these authors')
    st.pills(label = 'Find out more about the authors', options = set(st.session_state.suggestions), default = None, selection_mode="single", label_visibility='hidden', key = 'auto_query', on_change=run_graph)