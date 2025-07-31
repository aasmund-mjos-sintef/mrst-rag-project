import streamlit as st
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout = "wide", page_title = "MRST Assistant", page_icon = "images/mrst_logo.png")

mrst_logo, _, title, _, sintef_logo = st.columns([2, 4, 5, 4, 2])

with mrst_logo:
    st.image("images/mrst_logo.webp")

with title:
    st.markdown("# MRST Virtual Assistant")

with sintef_logo:
    st.image("images/sintef_logo.png")

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

if "query_suggestions" not in st.session_state:
    st.session_state.query_suggestions = []

if "auto_query" not in st.session_state:
    st.session_state.auto_query = ""

if "auto_author" not in st.session_state:
    st.session_state.auto_author = ""

if "c_fig" not in st.session_state:
    st.session_state.c_fig = ""

if "c_name" not in st.session_state:
    st.session_state.c_name = []

st.markdown("#### Hi, I am an assistant made by SINTEF for the Matlab Reservoir Simulation Toolbox. I can assist you by guiding you to which MRST developers you should contact based on your specific problem, and where in the MRST textbooks you might be able to get help regarding your problem.")

query, code_query, button = st.columns([6,6,1])
with query:
    query = st.text_area(label = "Query", key = "query", height = 300, label_visibility="hidden", placeholder="Please write down any problems you might have")
with code_query:
    code_query = st.text_area(label = "Code Query", key = "code_query", height = 300, label_visibility="hidden", placeholder="Please write down any matlab code for context")

from mrst_competence_query import graph
import io

def run_graph(state: graph.State = None):

    response_area.markdown("#### Generating answer...")

    if not state:

        if query != "" or code_query != "":

            state = graph.State(query = st.session_state.query,
                        code_query = st.session_state.code_query,
                        start_node = "InformationNode",
                        clustering = st.session_state.clustering,
                        github = st.session_state.github,
                        chapter_images = st.session_state.chapter_images,
                        text_answer = st.session_state.text_answer)
        
        else:
            return
        
    else:
        st.session_state.query = state.get('query')
        st.session_state.code_query = state.get('code_query')

    state = graph.invoke_graph(state)

    st.session_state.suggestions = state.get('suggestions')
    st.session_state.query_suggestions = state.get('query_suggestions')

    total_response = ""

    author_response = state.get('author_response')
    if author_response:
        total_response += f" \n\n### Relevant information about {", ".join(state.get('query_description').authors)}  \n\n"
        total_response += author_response

    book_response = state.get('book_response')
    if book_response:
        total_response += "\n\n### Relevant information in the MRST textbooks \n\n"
        total_response += book_response

    paper_response = state.get('paper_response')
    if paper_response:
        total_response += "\n\n### Relevant information in the MRST papers \n\n"
        total_response += paper_response

    relevance_score = state.get('relevance_score')
    if relevance_score: 
        if len(relevance_score.keys()):
            authors = sorted(list(zip(relevance_score.keys(), relevance_score.values())), key = lambda x: x[1])
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

    if figures:
        for c_info, fig in zip(chapter_info, figures):

            buf = io.StringIO()
            fig.savefig(buf, format = 'svg', facecolor = "#faf9f7")
            image = buf.getvalue()
            images.append((c_info, image))
            buf.close()

    st.session_state.figures = images

    c_fig = state.get('c_fig')
    if c_fig != None:
        buf = io.StringIO()
        c_fig.savefig(buf, format = 'svg', facecolor = "#faf9f7")
        image = buf.getvalue()
        st.session_state.c_fig = image
        st.session_state.c_name = state.get('c_name')
        buf.close()
    else:
        st.session_state.c_fig = ""

def reset_func():
    st.session_state.query = ""
    st.session_state.code_query = ""

def author_pills_callback():
    if st.session_state.get('auto_author'):
        state = graph.State(
            query = f"Give me a brief summary of what work {st.session_state.get('auto_author')} does",
            code_query= "",
            query_description = graph.QueryDescription(
                keywords = [],
                authors=[st.session_state.get('auto_author')],
                problem_description=""),
            start_node="RetrieveAuthorNode",
            clustering=st.session_state.clustering,
            github=st.session_state.github,
            chapter_images=st.session_state.chapter_images,
            text_answer=st.session_state.text_answer)
        run_graph(state)

def query_pills_callback():
    if st.session_state.get('auto_query'):
        state = graph.State(
            query = st.session_state.get('auto_query'),
            code_query= "",
            start_node="InformationNode",
            clustering=st.session_state.clustering,
            github=st.session_state.github,
            chapter_images=st.session_state.chapter_images,
            text_answer=st.session_state.text_answer)
        run_graph(state)

def create_suggestions():

    if len(st.session_state.query_suggestions):
        st.markdown('#### Query Suggestions')
        st.pills(label = 'Find out more about these topics',
                 options = st.session_state.query_suggestions,
                 default = None,
                 selection_mode="single",
                 label_visibility='hidden',
                 key = 'auto_query',
                 on_change=query_pills_callback)

    if len(st.session_state.suggestions):
        st.markdown('#### Suggested Authors')
        st.pills(label = 'Find out more about the authors',
                 options = st.session_state.suggestions,
                 default = None,
                 selection_mode="single",
                 label_visibility='hidden',
                 key = 'auto_author',
                 on_change=author_pills_callback)

with button:
    st.button(label = "Generate", on_click=run_graph, type = "primary")
    st.button(label = "Reset", on_click=reset_func)
    st.checkbox(label = "Cluster", key = "clustering", value=True, help = "If you want to cluster the MRST papers based on their content, check this box. This will take a bit longer to run.")
    st.checkbox(label = "Chapter Images", key = "chapter_images", value=True, help="If you want to show a graph over the relevant chapters in the MRST textbooks, check this box. This will take a bit longer to run.")
    st.checkbox(label = "Text Answer", key = "text_answer", value=True, help="If you want to get a text answer to your query, check this box. This will take a bit longer to run.")
    st.checkbox(label = "Git Hub", key = "github", value=False, help="If you want to retrieve relevant github commits in the MRST repository, check this box. This will take a bit longer to run.")

response_area = st.markdown(st.session_state.response)

if st.session_state.response:
    st.divider()

papers, github, _ = st.columns([6,6,1])

with papers:

    if len(st.session_state.authors):
        st.markdown("### Based on the retrieved MRST papers, I would recommend reaching out to: ")

    for a, s in st.session_state.authors:
        author_box, score_box = st.columns([1,1])
        with author_box:
            st.markdown(f'**{a}**')
        with score_box:
            st.markdown(f'With a relevance score of :orange[{str(round(s))}]')

with github:

    if len(st.session_state.github_authors):
        st.markdown("### Based on the relevant commits in the MRST github repository, I would recommend reaching out to: ")

    for a, s in st.session_state.github_authors:
        author_box, score_box = st.columns([1,1])
        with author_box:
            st.markdown(f'**{a}**')
        with score_box:
            st.markdown(f'With :orange[{str(s)}] commits in retrieved folders')

if len(st.session_state.authors):
    st.divider()

if bool(st.session_state.c_fig):

    st.markdown("### Map over 2D Reduced Vizualization of Relevant Papers")

    fig_box, _, select_box, _, suggestion_box, _ = st.columns([5,1,3,1,4,1])

    with fig_box:

        components.html("""
    <div style="
        padding: 10px;
        background-color: #faf9f7;
        display: inline-block;
    ">
        <div id="svg-container" style="width: auto; height: auto; overflow: auto;">
            """ + st.session_state.c_fig + """
        </div>
        <script src="https://unpkg.com/@panzoom/panzoom/dist/panzoom.min.js"></script>
        <script>
            var elem = document.getElementById('svg-container');
            var panzoom = Panzoom(elem, { maxScale: 5, minScale: 1});
            elem.addEventListener('wheel', panzoom.zoomWithWheel);
        </script>
    </div>
    """, width = int(600), height = int(400))
        
    with select_box:

        st.markdown("#### Cluster Names")

        for cluster, name, description in st.session_state.c_name:
            color, text = st.columns([1,5])
            with color:
                st.markdown(f'<div style="width: 40px; height: 40px; background-color: {graph.cluster_to_color.get(cluster, "#2B2929")}; border-radius: 50%;"></div>', unsafe_allow_html=True)
            with text:
                st.button(label = name,
                          key = name,
                          on_click=run_graph,
                          args = (graph.State(
                            query = name,
                            code_query = "",
                            start_node = "InformationNode",
                            clustering = st.session_state.clustering,
                            github = st.session_state.github,
                            chapter_images = st.session_state.chapter_images,
                            text_answer=st.session_state.text_answer),),
                          help = description)

    with suggestion_box:
        create_suggestions()
else:
    create_suggestions()

if bool(st.session_state.figures):
    st.divider()
    st.markdown("### Relevant Textbook Chapters")

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
    
    st.markdown(f"Map over chapter {c_info[0]} in {graph.book_to_url.get(c_info[1])} by {", ".join(c_info[2])}. A green node means that I found relevant content in that chapter. You can zoom in by scrolling.")