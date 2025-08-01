print("Loading graph.py...")
print("Importing libraries...")

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List, Literal
from sentence_transformers import SentenceTransformer
from importlib import resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkx import DiGraph
from networkx import draw as nx_draw
from networkx.drawing.nx_agraph import to_agraph
from matplotlib.figure import Figure
from langgraph.prebuilt import create_react_agent
from collections import Counter
from mrst_competence_query.classes.GitAgent import GitAgent
from mrst_competence_query.tools.sintef_search_tool import web_search_mrst
from hdbscan import HDBSCAN
from umap import UMAP

import warnings
warnings.filterwarnings('ignore')

from re import findall
from nltk.corpus import stopwords
from nltk import download as nltk_download
nltk_download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

from dotenv import load_dotenv
import os
load_dotenv()
langchain_openai_api_key = os.getenv("LANGCHAIN_OPENAI_API_KEY")

"""
Helper classes -- Called with client.with_structured_output(ClassName)
"""

class QueryDescription(BaseModel):
    keywords: List[str] = Field(description = "Keywords related to the users query")
    problem_description: str = Field(description = "The users problem compacted into one sentence")
    authors: List[str] = Field(description = "The authors mentioned in the users query")

class CodingKeyWords(BaseModel):
    keywords: List[str] = Field(description = "Key code object names or code words/functions etc based on the provided code.")

class SpecificScore(BaseModel):
    specific_scores: List[int] = Field(description = "List of specific_score is a number from 1 to 10 defining how specific or general the keywords are related to the scientific field of reservoir simulation. 10 is extremely specific, and 1 is extremely general")

class Authors(BaseModel):
    authors: List[str] = Field(description = "List of the authors mentioned in the users query. Has to be specifically mentioned by the user!")

class ClusterDescription(BaseModel):
    name: List[str] = Field(description = "List of the cluster names")
    description: List[str] = Field(description = "List of the cluster descriptions")

class QuerySuggestions(BaseModel):
    suggestions: List[str] = Field(description = "List of possible suggestions for further queries")

tools = [web_search_mrst]

print("Initializing AI clients and vector embedding model...")

nano_client = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0, openai_api_key=langchain_openai_api_key)
mini_client = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, openai_api_key=langchain_openai_api_key)
tool_client = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=langchain_openai_api_key)
tool_agent = create_react_agent(model = tool_client, tools = tools, response_format = QueryDescription)
vector_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
specter_embedding_model = SentenceTransformer("allenai-specter")
specter = False

print("Initializing Graph...")

"""
State
"""

class State(TypedDict):
    start_node: str
    query: str
    code_query: str
    coding_keywords: List[str]
    query_description : QueryDescription
    book_df: pd.DataFrame
    code_df: pd.DataFrame
    relevant_papers_df: pd.DataFrame
    book_response: str
    author_response: str
    paper_response: str
    suggestions: set
    query_suggestions: set
    figures: List[Figure]
    c_fig: Figure
    c_name: List[tuple[int, str, str]]
    chapter_info: tuple[int, str, List[str]]
    book_relevance_score: dict[str, float]
    authors_relevance_score: dict[str, float]
    relevance_score: dict[str, float]
    github_authors_relevance_score: dict[str, float]
    cosine_dict: dict[str, tuple[float, float]]
    clustering: bool
    github: bool
    chapter_images: bool
    text_answer: bool

"""
Helper objects
"""

cluster_to_color ={
-1:"#000000",
0:"#FF0000",
1:"#0000FF",
2:"#008000",
3:"#800080",
4:"#FFC0CB",
5:"#FFA500",
6:"#FFFF00",
7:"#808080"
}

relevant_node_color = "#eea46b"
unrelevant_node_color = "#ebe4d8"

advanced_section_to_author = {
    1: ["Runar L. Berge", "Øystein S. Klemetsdal", "Knut-Andreas Lie"],
    2: ["Mohammed Al Kobaisi", "Wenjuan Zhang"],
    3: ["Øystein S. Klemetsdal", "Knut-Andreas Lie"],
    4: ["Knut-Andreas Lie", "Olav Møyner"],
    5: ["Olav Møyner"],
    6: ["Olav Møyner"],
    7: ["Xin Sun", "Knut-Andreas Lie", "Kai Bao"],
    8: ["Olav Møyner"],
    9: ["Daniel Wong", "Florian Doster", "Sebastian Geiger"],
    10: ["Olufemi Olorode", "Bin Wang", "Harun Ur Rashid"],
    11: ["Rafael March", "Christine Maier", "Florian Doster", "Sebastian Geiger"],
    12: ["Marine Collignon", "Øystein S. Klemetsdal", "Olav Møyner"],
    13: ["Jhabriel Varela", "Sarah E. Gasda", "Eirik Keilegavlen", "Jan Martin"],
    14: ["Odd Andersen"]
}

introduction_section_to_author = {
    1: ["Knut-Andreas Lie"],
    2: ["Knut-Andreas Lie"],
    3: ["Knut-Andreas Lie"],
    4: ["Knut-Andreas Lie"],
    5: ["Knut-Andreas Lie"],
    6: ["Knut-Andreas Lie"],
    7: ["Knut-Andreas Lie"],
    8: ["Knut-Andreas Lie"],
    9: ["Knut-Andreas Lie"],
    10: ["Knut-Andreas Lie"],
    11: ["Knut-Andreas Lie"],
    12: ["Knut-Andreas Lie"],
    13: ["Knut-Andreas Lie"],
    14: ["Knut-Andreas Lie"],
    15: ["Knut-Andreas Lie"],
}

advanced_book = "Advanced Book"
introduction_book = "Introduction Book"

book_to_url = {
    advanced_book: "textbook [Advanced Modeling with the MATLAB Reservoir Simulation Toolbox](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/7AC2425C73F6F729DB88DB1A504FA1E7/9781316519967AR.pdf/Advanced_Modeling_with_the_MATLAB_Reservoir_Simulation_Toolbox.pdf?event-type=FTLA)",
    introduction_book: "textbook [An Introduction to Reservoir Simulation Using MATLAB/GNU Octave](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F48C3D8C88A3F67E4D97D4E16970F894/9781108492430AR.pdf/An_Introduction_to_Reservoir_Simulation_Using_MATLAB_GNU_Octave.pdf?event-type=FTLA)"
}

"""
Helper functions
"""

def load_dataframe(filename: str) -> pd.DataFrame:
    """
    Returns pandas Dataframe from pickle file
    """
    with resources.files("mrst_competence_query.datasets").joinpath(filename).open("rb") as f:
        return pd.read_pickle(f)

def split_by_more(string: str, chars: List[str] = []) -> List[str]:
    """
    Almost the same as str.split, only splits by multiple characters,
    and does so by replacing chars[i] with chars[i+1] until the last char,
    which is used to split the string
    """
    if chars == []:
        return string.split()
    else:
        for method, next_method in zip(chars, chars[1:]):
            string = next_method.join(string.split(method))
        return string.split(chars[-1])

def name_in(name: str, name_list: List[str]) -> bool:
    """
    Checkis if name is in name_list.
    Two names are considered equal if the first letter of the first word in each name is equal,
    and if the last word in each name is equal.
    """
    if name in name_list:
        return True
    
    "Second, check if first name and last name is equal -> If so, regarded as equal"
    for name_2 in name_list:
        name_split = split_by_more(name, ["-","."," "])
        name_2_split = split_by_more(name_2, ["-","."," "])
        if len(name_split)>=2 and len(name_2_split)>=2:
            if name_split[0][0] == name_2_split[0][0] and name_split[-1] == name_2_split[-1]:
                return True
    return False

def new_title(title: str, n_chars: int = 10) -> str:
    """
    Splits the title into multiple lines
    When chars on one line exceedds n_chars,
    next word will be placed on the next line
    """
    t_split = title.split()
    new_title = ""
    current = ""
    for word in t_split:
        current += " "+word
        if len(current) > n_chars:
            new_title += current
            new_title += "\n"
            current = ""
    new_title += current
    return new_title

def node_size(number_of_paths_out_of_node: int) -> int:
    """
    Calculates Node Size based on number of paths out of node
    """
    if number_of_paths_out_of_node:
        return 5000
    else:
        return 1500

def font_size(number_of_paths_out_of_node: int) -> int:
    """
    Calculates Font Size based on number of paths out of node
    """
    if number_of_paths_out_of_node:
        return 5
    else:
        return 3
    
def get_bigram_freq(text: str) -> Counter:
    """
    Returns Counter object of bigrams in text.
    """
    words = findall(r'\w+', text.lower()) 
    words = [word for word in words if word not in stop_words and len(word) > 2]

    bigrams = zip(words, words[1:])
    bigram_list = [' '.join(bigram) for bigram in bigrams]

    return Counter(bigram_list)

def get_top_bigrams(df: pd.DataFrame) -> List[tuple[str, int]]:
    """
    Gathers top bigrams from df['content']
    Returns list of tuple with structure (bigram, count)
    """
    total_text = ""
    for c in df['content'].tolist():
        total_text += c

    counter = get_bigram_freq(total_text)
    return counter.most_common(10)

def generate_book_graph_figure(chapter: int, book: str, sections: set[tuple[int, int, int]]) -> Figure:
    """
    Generates chapter graphs showing the structure of the chapter in the book.
    sections are relevant chapters. These will be highlighted with orange colour
    Requires a way to generate pygraphviz graph from networkx.DiGraph
    Returns matplotlib figure.
    """
    book_df = load_dataframe("book_embeddings.pkl")
    df = book_df[(book_df['file_type'].isin([book])) & (book_df['0'].isin([chapter]))]
    t = df['title'].tolist()
    s = df['sections'].tolist()

    titles = [s[i] +": "+ t[i] for i in range(len(df))]
    first = df['1'].tolist()
    second = df['2'].tolist()
    G = DiGraph()
    G.add_nodes_from([new_title(t) for t in titles])
    for i in range(len(df)):
        if second[i] != 0:
            for j in range(len(df)):
                if second[j] == 0 and first[i] == first[j]:
                    G.add_edge(new_title(titles[j]), new_title(titles[i]))
                    break
        else:
            if first[i]!=0:
                for j in range(len(df)):
                    if first[j] == 0:
                        main_title = new_title(titles[j])
                        G.add_edge(main_title, new_title(titles[i]))
                        break

    
    node_sizes = [node_size(G.out_degree(n)) for n in G.nodes()]
    font_sizes = [font_size(G.out_degree(n)) for n in G.nodes()]
    node_colors = []
    for i in range(len(df)):
        if (chapter, first[i], second[i]) in sections:
            node_colors.append(relevant_node_color)
        else:
            node_colors.append(unrelevant_node_color)

    A = to_agraph(G)

    for i, n in enumerate(G.nodes()):
        r = np.sqrt(node_sizes[i]/100)
        A.get_node(n).attr.update(width=r, height=r, fixedsize="true")

    A.graph_attr.update(ranksep='1', nodesep='1')
    A.layout(prog="dot")

    pos = {n: tuple(map(float, A.get_node(n).attr["pos"].split(','))) for n in G.nodes()}

    for i in range(len(df)):
        if first[i]==0 and second[i]==0:
            node_sizes[i] = 20000
            font_sizes[i] = 10
            break

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.set_facecolor('#faf9f7')
    ax.set_facecolor('#faf9f7')
    nx_draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors, ax=ax)
    for i, (n, (x, y)) in enumerate(pos.items()):
        if first[i] == 0 and second[i] == 0:
            ax.text(x, y, n, fontsize=font_sizes[i], ha='center', va='center')
        else:
            ax.text(x, y, n, fontsize=font_sizes[i], ha='center', va='center')

    max_radius = max(np.sqrt(s / 10) for s in node_sizes)
    x_vals, y_vals = zip(*pos.values())

    ax.set_xlim(min(x_vals) - 3*max_radius, max(x_vals) + 3*max_radius)
    ax.set_ylim(min(y_vals) - 3*max_radius, max(y_vals) + 7*max_radius)
    
    ax.set_axis_off()
    return fig

def umap_reduce(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame, creates a manifold on the 768-dim vectors using the UMAP algorithm to reduce them to 50 dimensions.
    It returns None if the number of vectors is too low to create a manifold.
    Else, it returns the DataFrame with the new column 'embedding_umap' containing the 50-dim vectors.
    """

    try:
        umap_reducer = UMAP(n_neighbors=10, min_dist=0, n_components=50, random_state=42)
        umap_embeddings = umap_reducer.fit_transform(filtered_df['embedding'].tolist())
        filtered_df['embedding_umap'] = list(umap_embeddings)
        return filtered_df
    except Exception as e:
        return pd.DataFrame()

def hdbscan_cluster(reduced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with 50-dim vectors, clusters the vectors using the HDBSCAN algorithm.
    Input is the reduced_df from umap_reduce(df). If reduced_df is None, returns empty df
    It returns the DataFrame with a new column 'cluster' containing the assigned cluster for each row.
    """

    if reduced_df.empty:
        return reduced_df

    if len(reduced_df) < 20:
        min_cluster_size = 3
    else:
        min_cluster_size = int(len(reduced_df) / 10)+1

    embeddings = reduced_df['embedding_umap'].tolist()
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=5,
                                    cluster_selection_epsilon=0.6,
                                    metric='euclidean', 
                                    cluster_selection_method='eom', 
                                    allow_single_cluster=False,
                                    gen_min_span_tree=True,
                                    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    reduced_df['cluster'] = cluster_labels

    return reduced_df

def cluster_names(clustered_df: pd.DataFrame) -> List[tuple[int, str, str]]:
    """
    Given a clustered df from hdbscan_cluster(umap_reduce(df)),
    returns a list of (cluster_nr, name, description) for each cluster,
    where cluster_nr is the integer nr. for a cluster,
    name and description are AI generated descriptions of the clusters
    based on the most frequent bigrams in the clusters abstracts
    """
    different_clusters = set(clustered_df['cluster'].tolist())
    top_words = []
    for c in different_clusters:
        c_df = clustered_df[clustered_df['cluster'] == c]
        top_words.append(",".join([bigram for bigram, counter in get_top_bigrams(c_df)]))
    n = len(different_clusters)
    msg = [{"role": "system", "content": f"""
            You are going to create a name and a 1 sentence long cluster description for each of the scientific article clusters
            identified by the following top bigrams, so a total of {n}. Your mission is to create names and descriptions that help highlight the differences between the clusters.
            
            -{"\n\n-".join(top_words)}"""}]
    cluster_description = nano_client.with_structured_output(ClusterDescription).invoke(msg)
    name = cluster_description.name
    description = cluster_description.description
    return list(zip(different_clusters, name, description))

def get_paper_response_if_not_cluster(query: str, df: pd.DataFrame) -> str:
    """
    Given the users query, and a filtered/sorted df,
    returns an AI generated text about relevant authors
    """
    context = "\n + ""\n\n".join([f" title: {t}\n authors: {", ".join(a)}\n content: {c}" for t,a,c in zip(df['titles'], df['authors'], df['content'])])
    msg = [{"role": "system", "content": f"""
            You are going to guide the user to the authors best suited to help with their problem.
            State who is relevant to contact about different subtopics presented in the context related to the users query.
            Do not try and solve the users problem.
            Context:{context}"""}, {"role":"user", "content": query}]
    return nano_client.invoke(msg).content

def get_cluster_response(query: str, df: pd.DataFrame) -> str:
    """
    Given the users query, and a df consisting of papers in one cluster
    (you should not input all the documents in a cluster, but rather take the top 10 or 5 or something),
    returns an AI generated text about relevant authors. 
    """
    context = "\n + ""\n\n".join([f" title: {t}\n authors: {", ".join(a)}\n content: {c}" for t,a,c in zip(df['titles'], df['authors'], df['content'])])
    msg = [{"role": "system", "content": f"""
            You are going to guide the user to the authors best suited to help with their problem.
            State who is relevant to contact about different subtopics presented in the context related to the users query.
            Do not try and solve the users problem. Your answer should be a short paragraph.
            Context:{context}"""}, {"role":"user", "content": query}]
    return nano_client.invoke(msg).content

"""
Nodes
"""

def InformationNode(state: State) -> State:
    """
    Generates keywords, a problem description, and authors from a users query
    """
    query = state.get("query")

    print("--Information Node--")

    msg = [('system', f"""
    You are an assistant for the Matlab Reservoir Toolbox developed by SINTEF.
    You are going to extract keywords and a problem description from the user query.
    The keywords should help distinguish different SINTEF researchers MRST expertize fields,
    so you should be extremely specific when generating keywords.
            
    For example, if the query is:
        - Where can I learn more about chemical eor?
    Possible keywords are (even though you should generate more):
        - chemical eor
        - chemical enhanced oil recovery
        - polymer flooding
            
    The problem description should be a short sentence describing the users problem.
    You can use the tool 'web_search_mrst' to get relevant content about mrst modules,
    and further links from the mrst webpage.
    Only use the tool if you're sure the link is relevant.
    Here's a list of possible start links, if you are going to use the tool, choose the most relevant one:

        {web_search_mrst.invoke(input = 'https://www.sintef.no/projectweb/mrst/modules/')[1]}

    """),
    ("user", query)]

    query_description = tool_agent.invoke({"messages": msg}).get('structured_response')

    problem_description = query_description.problem_description
    keywords = query_description.keywords

    authors = nano_client.with_structured_output(Authors).invoke([{"role": "system", "content": "You are going to extract specific SINTEF researchers from a users query. In other words, only return a list if there is a human name in the query"}, {"role": "user", "content": query}]).authors
    authors = [a for a in authors if "sintef" not in a.lower()]

    expensive = True
    if expensive == True:
        if len(query) < 500:    # Since we use an expensive model, we only allow smaller prompts into the llm

            keywords_string = ""
            for k in keywords:
                keywords_string += "-"+k
                keywords_string += "\n"

            n = len(keywords)

            sys_msg = f"""You are going to determine how specific these keywords are.
            The keywords are all related to the scientific field of reservoir simulation.
            For example, keywords like 'reservoir simulation', 'numerical mathematics' should always have a low score like 1 or 2, since they are very general in this field. 
            For example, a keyword like 'chemical enhanced oil recovery' should be very high if the query is 'What can you tell me about chemical eor'
            Generate scores for all keywords, so a total of {n}.
            Query:\n{query}
            
            Keywords:\n {keywords_string}"""

            specific_score = []

            i = 0
            while len(specific_score) != n and i<3:
                i+=1
                specific_score = mini_client.with_structured_output(SpecificScore).invoke([{"role": "system", "content": sys_msg}]).specific_scores

            if i < 3:
                keywords = [keywords[j] for j in range(n) if specific_score[j]>6]

    query_description = QueryDescription(
        keywords=keywords,
        problem_description=problem_description,
        authors=authors
    )

    return {"query_description": query_description}

def SearchMRSTModulesNode(state: State) -> State:
    """
    Only here to vizualize a node in the graph
    """
    return {}

def RetrieveBookNode(state: State) -> State:
    """
    Retrieves Relevant Book Sub Chapters
    """
    print("--Retrieve Book Node--")

    df = load_dataframe('book_embeddings.pkl')
    df = df[df['file_type'].isin(['Advanced Book'])]
    query_description = state.get('query_description')
    vector = vector_embedding_model.encode(query_description.keywords + [query_description.problem_description])
    embeddings = np.array(df['embedding'].tolist())
    dot_prod = np.einsum('ij,kj->ki', vector, embeddings)
    vec_prod = np.einsum('i,k->ki',np.linalg.norm(vector, axis = -1),np.linalg.norm(embeddings, axis = -1))
    cosines = dot_prod/vec_prod
    df['cosine'] = np.max(cosines, axis = -1)
    book_df = df[df['cosine'] >= 0.65]

    book_relevance_score = {}

    for authors in book_df['authors']:
        for a in authors:
            a_split = split_by_more(a, ['-','.',' '])
            a_formatted = a_split[0][0] + "." + a_split[-1]
            book_relevance_score[a_formatted] = book_relevance_score.get(a_formatted,0) + 5

    return {"book_df": book_df, "book_relevance_score": book_relevance_score}

def GenerateBookNode(state: State) -> State:
    """
    Retrieves an AI generated text about relevant authors in the relevant book-chapters
    """
    print("--Generate Book Node--")

    query = state.get('query')
    df = state.get("book_df")

    content = df['content'].tolist()
    authors = df['authors'].tolist()
    sections = df['sections'].tolist()
    titles = df['title'].tolist()
    book = df['file_type'].tolist()
    chapters = df['0'].tolist()
    first = df['1'].tolist()
    second = df['2'].tolist()
    zipped_book = set(zip(chapters, book))
    s = set(zip(chapters, first, second))

    chapter_info = []
    figures = []

    for c, b in zipped_book:
        if b == "Advanced Book":
            rel_authors = advanced_section_to_author.get(c)
        else:
            rel_authors = introduction_section_to_author.get(c)
        chapter_info.append((c,b,rel_authors))
        if state.get('chapter_images', True):
            figures.append(generate_book_graph_figure(c, b, sections=s))

    book_response = ""
    if state.get('text_answer', True):

        df = df.sort_values(by = 'cosine', ascending=False).head(5)

        context = "\n\n".join(["Section " + sections[i] + " in book: " + book[i] + "\n Title" + titles[i] + "\n Authors: " + ",\t".join(authors[i]) + "\n Content: " + content[i] for i in range(len(df))])
        msg = [{"role": "system", "content": f"""You are the Generator in a RAG application.
                You are going to guide the user to which section in the book could be relevant for their question,
                and state which authors they can reach out to for questions regarding their query.
                You are going to call the book Advanced Book,
                Do not mention the titles of the chapters.
                Simply state which researchers work with the relevant subtopics,
                and therefore who the user should contact.
                The context is relevant sections in the Advanced Book
                \n Context:\n {context}"""}, {"role": "user", "content": query}]
        
        book_response = nano_client.invoke(msg).content.replace(advanced_book, book_to_url.get(advanced_book, "")).replace(introduction_book, book_to_url.get(introduction_book, ""))

    return {"book_response": book_response,
            "figures": figures,
            "chapter_info": chapter_info}

def RetrieveAuthorNode(state: State) -> State:
    """
    Retrieves relevant chapters and papers based on authors mentioned in the users query
    """

    print("--Retrieve Author Node--")

    df = load_dataframe('book_embeddings.pkl')
    df = df[df['file_type'].isin(['Advanced Book'])]
    authors_names = state.get('query_description').authors
    book_df = df[df['authors'].apply(lambda x: bool([True for a in x if name_in(a, authors_names)]))]

    df = load_dataframe('mrst_abstracts_embedding.pkl')
    relevant_papers_df = df[df['authors'].apply(lambda x: bool([True for a in x if name_in(a, authors_names)]))]

    return {"book_df": book_df, "relevant_papers_df": relevant_papers_df}

def GenerateAuthorNode(state: State) -> State:
    """
    Returns an AI generated text about relevant chapters and papers from the authors mentioned in the users query
    """

    print("--Generate Author Node--")

    df = state.get("book_df")
    query = state.get("query")

    author_response = ""
    figures = None
    chapter_info = None

    if len(df) > 0:

        authors = df['authors'].tolist()
        book = df['file_type'].tolist()
        chapters = df['0'].tolist()
        first = df['1'].tolist()
        second = df['2'].tolist()
        sec = df['sections']
        zipped_book = set(zip(chapters, book))
        s = set(zip(chapters, first, second))

        figures = []
        chapter_info = []

        for c, b in zipped_book:
            if b == "Advanced Book":
                rel_authors = advanced_section_to_author.get(c)
            else:
                rel_authors = introduction_section_to_author.get(c)

            chapter_info.append((c,b,rel_authors))
            figures.append(generate_book_graph_figure(c, b, sections=s))

        if state.get('text_answer', True):

            titles = df['title'].tolist()
            authors = [", ".join(a) for a in df['authors']]
            context = "\n\n".join([" title: " + k.strip() + " " + i+"\n Authors: "+j for i,j,k in zip(titles, authors, sec)])
            msg = [{"role": "system", "content": "You are the Generator in RAG application. "+
                "You are going to answer the users query, based on the context given. If there are multiple researchers mentioned in the users query, "+
                "answer the query seperately for each researcher."+
                "The context given is titles of chapters in the 'Advanced Book', and their respective authors. When mentioning the book write 'Advanced Book' as the name"+
                "\n Context:\n" + context}, {"role": "user", "content": query}]
            
            author_response += "#### MRST Books \n\n"
            author_response += nano_client.invoke(msg).content
            author_response += "\n\n"

    df = state.get("relevant_papers_df")

    if len(df) > 5 and state.get('text_answer', True):

        titles = df['titles'].tolist()
        authors = [", ".join(a) for a in df['authors']]
        context = "\n\n".join([" title: " + i+"\n Authors: "+j for i,j in zip(titles, authors)])
        msg = [{"role": "system", "content": "You are the Generator in RAG application. "+
            "You are going to answer the users query, based on the context given. If there are multiple researchers mentioned in the users query, "+
            "answer the query seperately for each researcher. "+
            "The context given is titles of scientific articles, and their respective authors. "+
            "\n Context:\n" + context}, {"role": "user", "content": query}]
        
        author_response += "#### MRST Papers \n\n"
        author_response += nano_client.invoke(msg).content
        author_response += "\n\n"
    
    elif len(df) > 0 and state.get('text_answer', True):

        titles = df['titles'].tolist()
        authors = [", ".join(a) for a in df['authors']]
        abstracts = df['content'].tolist()
        context = "\n\n".join([" title: " + i+"\n Authors: "+j + "\n Abstract: "+k for i,j,k in zip(titles, authors, abstracts)])
        msg = [{"role": "system", "content": "You are the Generator in RAG application. "+
            "You are going to answer the users query, based on the context given. If there are multiple researchers mentioned in the users query, "+
            "answer the query seperately for each researcher. "+
            "The context given is titles of scientific articles, their respective authors and the abstracts. "+
            "\n Context:\n" + context}, {"role": "user", "content": query}]
        
        author_response += "#### MRST Papers \n\n"
        author_response += nano_client.invoke(msg).content
        author_response += "\n\n"

    if author_response == "" and state.get('text_answer', True):
        author_response = "Sorry, but I couldn't find any relevant information"

    else:
        return {"author_response": author_response.replace(advanced_book, book_to_url.get(advanced_book, "")).replace(introduction_book, book_to_url.get(introduction_book, "")), "figures": figures, "chapter_info": chapter_info}

def SearchAndEvaluateNode(state: State) -> State:
    """
    Retrieves relevant abstracts to the user's query, reranks the papers and generates text about relevant authors based on the reranked papers
    """

    print("--Search and Evaluate Node Abstracts--")

    if specter:

        df = load_dataframe('mrst_abstracts_embedding_specter.pkl')
        query_description = state.get('query_description')

        keywords = query_description.keywords
        keywords = keywords if len(keywords) else [query_description.problem_description]
        keyword_embeddings = vector_embedding_model.encode(keywords)

        vector = np.array(specter_embedding_model.encode(query_description.keywords + [query_description.problem_description]))

        embeddings = np.array(df['embedding'].tolist())
        dot_prod = np.einsum('ij,kj->ki', vector, embeddings)
        vec_prod = np.einsum('i,k->ki',np.linalg.norm(vector, axis = -1),np.linalg.norm(embeddings, axis = -1))
        cosines = dot_prod/vec_prod
        df['cosine'] = np.max(cosines, axis = -1)

        threshold = np.max(cosines)-0.07
        sorted_df = df[df['cosine'] > threshold]
    
    else:
        df = load_dataframe('mrst_abstracts_embedding.pkl')
        query_description = state.get('query_description')

        keywords = query_description.keywords
        keywords = keywords if len(keywords) else [query_description.problem_description]
        keyword_embeddings = vector_embedding_model.encode(keywords)

        vector = vector_embedding_model.encode(query_description.keywords + [query_description.problem_description])

        embeddings = np.array(df['embedding'].tolist())
        dot_prod = np.einsum('ij,kj->ki', vector, embeddings)
        vec_prod = np.einsum('i,k->ki',np.linalg.norm(vector, axis = -1),np.linalg.norm(embeddings, axis = -1))
        cosines = dot_prod/vec_prod
        df['cosine'] = np.max(cosines, axis = -1)
        threshold = min([0.5, np.max(cosines)-0.2])

        sorted_df = df[df['cosine'] > threshold]

    if len(sorted_df) > 0:

        n_bigrams = 5

        top_bigrams = [[tup[0] for tup in get_bigram_freq(x).most_common(n_bigrams)] for x in sorted_df['content'].tolist()]

        top_bigrams_embedded = np.array([vector_embedding_model.encode(x) for x in top_bigrams])

        dot_prod = np.einsum('xnj,kj->xnk', top_bigrams_embedded, keyword_embeddings)
        vec_prod = np.einsum('xn,k->xnk',np.linalg.norm(top_bigrams_embedded, axis = -1),np.linalg.norm(keyword_embeddings, axis = -1))
        cosines = dot_prod/vec_prod

        sorted_df['avg_high_cosine'] = np.mean(np.max(cosines, axis = -1), axis = -1)
        sorted_df['score'] = [c + a_h_c for c, a_h_c in zip(sorted_df['cosine'], sorted_df['avg_high_cosine'])]

        authors_abstract_score  = {}
        authors_n_rel_abstracts = {}
        authors_total_n_papers  = {}
        authors_relevance_score = {}

        authors_set = set()

        for authors in df['authors'].tolist():
            for a in authors:
                authors_total_n_papers[a] = authors_total_n_papers.get(a, 0) + 1

        for authors, score in zip(sorted_df['authors'], sorted_df['score']):
            for a in authors:
                authors_set.add(a)
                authors_abstract_score[a] = authors_abstract_score.get(a, 0) + (1+score)**2-1
                authors_n_rel_abstracts[a] = authors_n_rel_abstracts.get(a, 0) + 1
        
        max_n_papers = 0
        for a in authors_set:
            if authors_total_n_papers[a] > max_n_papers:
                max_n_papers = authors_total_n_papers[a]

        gamma = 0.2

        for a in authors_set:
            authors_relevance_score[a] = 100*authors_abstract_score[a]/(authors_total_n_papers[a]*max_n_papers)**gamma

        c_fig = None
        c_name = None
        paper_response = ""

        if state.get('clustering'):

            clustered_df = hdbscan_cluster(umap_reduce(sorted_df))

            if not clustered_df.empty:
                embeddings_768 = clustered_df['embedding'].tolist()

                fig, ax = plt.subplots(figsize=(6, 4))
                fig.set_facecolor('#faf9f7')
                ax.set_facecolor('#faf9f7')
                clusters = clustered_df['cluster'].tolist()

                for_visual_umap_reducer = UMAP(n_neighbors=15, min_dist=0.1,  metric='euclidean', n_components=2, random_state=42)
                umap_embeddings = for_visual_umap_reducer.fit_transform(embeddings_768)
                x_umap = [embd[0] for embd in umap_embeddings]
                y_umap = [embd[1] for embd in umap_embeddings]
                ax.scatter(x_umap, y_umap, c=[cluster_to_color.get(x) for x in clusters])
                ax.set_axis_off()
                c_fig = fig
                c_name = cluster_names(clustered_df)

                if state.get('text_answer', True):

                    c_to_n = {}
                    for c, n, _ in c_name:
                        c_to_n[c] = n

                    different_clusters = set(clusters)
                    if len(different_clusters) > 3:
                        paper_response = get_paper_response_if_not_cluster(query = state.get('query'), df=sorted_df.sort_values(by = 'score', ascending = False).head(10))
                    else:
                        query = state.get('query')
                        for c in different_clusters:
                            paper_response += f"#### {c_to_n.get(c)}\n"
                            paper_response += get_cluster_response(query = query, df = clustered_df[clustered_df['cluster'] == c].sort_values(by = 'score', ascending = False).head(5))
                            paper_response += "\n\n"

            else:
                if state.get('text_answer', True):
                    paper_response = get_paper_response_if_not_cluster(query = state.get('query'), df=sorted_df.sort_values(by = 'score', ascending = False).head(10))

        else:
            if state.get('text_answer', True):
                paper_response = get_paper_response_if_not_cluster(query = state.get('query'), df=sorted_df.sort_values(by = 'score', ascending = False).head(10))

        return {"authors_relevance_score": authors_relevance_score, "relevant_papers_df": sorted_df, "c_fig": c_fig, "c_name": c_name, "paper_response": paper_response}

    else:
        return {}

def GitNode(state: State) -> State:
    """
    Retrieves relevant commits in relevant folders to the user's query and matlab query.
    Uses the commits to return relevant github authors.
    """

    print("--Git Node--")

    query = state.get('query')
    code_query = state.get('code_query')
    prompt = [{"role": "system", "content": "You are going to extract a minimum of 2 and a maximum of 10 code keywords related to the problem the user has based on the provided code and problem"},{"role":"user", "content": "problem:\n " + query + "\n\ncode:\n" + code_query}]
    coding_keywords = mini_client.with_structured_output(CodingKeyWords).invoke(prompt).keywords
    code_search = coding_keywords
    code_search_embeddings = np.array(vector_embedding_model.encode(code_search))
    df = load_dataframe("mrst_repository_embeddings.pkl")
    embeddings = np.array(df['embedding'].tolist())
    dot_prod = np.einsum("ni,ki->nk", code_search_embeddings, embeddings)
    norm_prod = np.einsum("n,k->nk", np.linalg.norm(code_search_embeddings, axis = -1), np.linalg.norm(embeddings, axis = -1))
    cosine = dot_prod/norm_prod
    df['cosine'] = np.max(cosine, axis = 0)
    sorted_df = df.sort_values(by = 'cosine', ascending = False).head(10)
    file_paths = sorted_df['file_path'].tolist()
    path_df = {}
    for fp in file_paths:
        fp_layers = fp.split('/')
        for i in range(1,len(fp_layers)):
            path = "/".join(fp_layers[:i])
            path_df[path] = path_df.get(path,0) + 1

    most_common_paths = sorted(list(zip(path_df.keys(), path_df.values())), key=lambda x: x[1])[::-1][:5]
    gitAgent = GitAgent()
    total_freq_dict = {}
    for p, n in most_common_paths:
        _, name = gitAgent.get_commit_frequency_numbers(p)
        for n in name.keys():
            total_freq_dict[n] = total_freq_dict.get(n,0) + name[n]

    return {"code_df": sorted_df, "coding_keywords": coding_keywords, "github_authors_relevance_score": total_freq_dict}

def SuggestionsNode(state: State) -> State:

    print("--Suggestions Node--")

    chapter_info = state.get('chapter_info')
    suggestions = set()
    if chapter_info != None:
        for _,_,rel_authors in chapter_info:
            for a in rel_authors:
                name_split = split_by_more(a, ["-","."," "])
                suggestions.add(name_split[0][0]+"."+name_split[-1])

    relevance_score = state.get('authors_relevance_score', None)
    if relevance_score != None:
        authors = sorted(list(zip(relevance_score.keys(), relevance_score.values())), key = lambda x: x[1])
        total_n_authors = len(authors)
        n = min([total_n_authors, 5])
        suggested_authors = authors[total_n_authors-n:]
        for a, s in suggested_authors:
            suggestions.add(a)

    total_context = "\n"
    total_context += state.get('paper_response', '')+"\n\n" if state.get('paper_response') else ""
    total_context += state.get('author_response', '')+"\n\n" if state.get('author_response') else ""
    total_context += state.get('book_response', '')+"\n\n" if state.get('book_response') else ""
    total_context += f'{"\n".join([f'{n}: {d}' for c, n, d in state.get('c_name', '')])}'+"\n\n" if state.get('c_name') else ""
    
    msg = [{"role": "system", "content": f"""You are a next query suggestion maker tool in the Matlab Reservoir Simulation Toolbox competence query developed by SINTEF.
            The users query has been used to generate answers, which is the context provided.
            Based only on the provided context, generate a list of of subtopics for further simulations.
            Create a maximum of 7 such suggestions. Each suggestion can maximum be 7 words.
            Context: {total_context}"""}, {"role": "user", "content": state.get('query')}]
    
    query_suggestions = set(nano_client.with_structured_output(QuerySuggestions).invoke(msg).suggestions)

    kappa = 0.7

    relevance_score = {}
    a_r = state.get('authors_relevance_score', {})
    b_r = state.get('book_relevance_score', {})
    
    if a_r and b_r:

        m_a = max(a_r.values())
        m_b = max(b_r.values())
        weight_b = kappa*m_a/m_b
        weight_a = 1

        for author in a_r.keys():
            relevance_score[author] = relevance_score.get(author, 0) + weight_a * a_r.get(author, 0)
        
        for author in b_r.keys():
            relevance_score[author] = relevance_score.get(author, 0) + weight_b * b_r.get(author, 0)
        
    elif a_r:
        for author in a_r.keys():
            relevance_score[author] = relevance_score.get(author, 0) + a_r.get(author, 0)

    elif b_r:
        for author in b_r.keys():
            relevance_score[author] = relevance_score.get(author, 0) + b_r.get(author, 0)

    return {"suggestions": suggestions, "query_suggestions": query_suggestions, "relevance_score": relevance_score}

"""
Routers
"""
def StartNodeRouter(state: State) -> Literal["InformationNode", "RetrieveAuthorNode"]:
    """
    Uses start_node to choose where to start the graph excecution
    """
    return state.get('start_node')

def RetrievalRouter(state: State) -> Literal["SearchMRSTModulesNode","RetrieveBookNode","RetrieveAuthorNode"]:
    """
    If authors are detected in the query_description, guides the program to the RetrieveAuthorNode
    """
    queryDescription = state.get('query_description')
    if queryDescription.authors:
        return "RetrieveAuthorNode"
    else:
        return "RetrieveBookNode"

def BookRouter(state: State) -> Literal["GenerateBookNode", "SearchAndEvaluateNode"]:
    """
    If relevant chapters are found, takes detour to GenerateBookNode
    """
    if len(state.get('book_df')):
        return "GenerateBookNode"
    else:
        return "SearchAndEvaluateNode"
    
def GitRouter(state: State) -> Literal["GitNode", "SuggestionsNode"]:
    """
    If user set github search (state.github) to True, guides the program to GitNode
    """
    if state.get('github'):
        return "GitNode"
    else:
        return "SuggestionsNode"

"""
Setting up the graph
"""

graph_builder = StateGraph(State)

graph_builder.add_node("InformationNode", InformationNode)
graph_builder.add_node("SearchMRSTModulesNode", SearchMRSTModulesNode)
graph_builder.add_node("RetrieveBookNode", RetrieveBookNode)
graph_builder.add_node("GenerateBookNode", GenerateBookNode)
graph_builder.add_node("SearchAndEvaluateNode", SearchAndEvaluateNode)
graph_builder.add_node("RetrieveAuthorNode", RetrieveAuthorNode)
graph_builder.add_node("GenerateAuthorNode", GenerateAuthorNode)
graph_builder.add_node("GitNode", GitNode)
graph_builder.add_node("SuggestionsNode", SuggestionsNode)

graph_builder.add_conditional_edges(START, StartNodeRouter, {"InformationNode": "InformationNode", "RetrieveAuthorNode": "RetrieveAuthorNode"})
graph_builder.add_conditional_edges("InformationNode", RetrievalRouter, {"SearchMRSTModulesNode": "SearchMRSTModulesNode", "RetrieveBookNode": "RetrieveBookNode", "RetrieveAuthorNode": "RetrieveAuthorNode"})
graph_builder.add_conditional_edges("RetrieveBookNode", BookRouter, {"GenerateBookNode":"GenerateBookNode", "SearchAndEvaluateNode":"SearchAndEvaluateNode"})
graph_builder.add_edge("GenerateBookNode", "SearchAndEvaluateNode")
graph_builder.add_edge("SearchMRSTModulesNode", "InformationNode")
graph_builder.add_conditional_edges("SearchAndEvaluateNode", GitRouter, {"GitNode": "GitNode", "SuggestionsNode": "SuggestionsNode"})
graph_builder.add_edge("RetrieveAuthorNode", "GenerateAuthorNode")
graph_builder.add_edge("GenerateAuthorNode", "SuggestionsNode")
graph_builder.add_edge("GitNode", "SuggestionsNode")
graph_builder.add_edge("SuggestionsNode", END)

program = graph_builder.compile()

def invoke_graph(state: State) -> State:
    """
    Invoke the graph
    """
    print("Invoking Graph!")
    return program.invoke(state)
    

def get_graph_vizualization(file_path='images/graph_vizualization.png'):
    """
    file_path must be <filename>.png
    Draw mermaid plot of the graph and save png to file_path
    """
    program.get_graph().draw_mermaid_png(output_file_path=file_path)

print("Graph has successfully been built, and is ready for use\n")