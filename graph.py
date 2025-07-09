from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List, Literal
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tools.sintef_search_tool import web_search_mrst

from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_openai_api_key = os.getenv("LANGCHAIN_OPENAI_API_KEY")

strong_client = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=langchain_openai_api_key)
weak_client = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key = langchain_openai_api_key)

vector_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

"""
Helper classes -- Called with client.with_structured_output(ClassName)
"""

class QueryDescription(BaseModel):
    keywords: List[str] = Field(description = "Keywords related to the users query")
    authors: List[str] = Field(description = "The authors mentioned in the users query")
    problem_description: str = Field(description = "The users problem compacted into one sentence")
    tools: bool = Field(description = "Wether or not should call tools")
    tools_input: str = Field(description = "The input to the tool")

class State(TypedDict):
    query: str
    query_description : QueryDescription
    attempts: int
    df: pd.DataFrame
    response: str
    figures: List[Figure]
    tools_calls: List[tuple[str, List[str]]]
    visited_links: List[str]
    n_tool_calls: int

"""
Helper functions
"""

def name_in(name: str, name_list: List[str]) -> bool:

    "First, simple check to see if exact name is listed in name_list"
    if name in name_list:
        return True
    
    "Second, check if first name and last name is equal -> If so, regarded as equal"
    for name_2 in name_list:
        name_split = name.split()
        name_2_split = name_2.split()
        if len(name_split)>=2 and len(name_2_split)>=2:
            if name_split[0] == name_2_split[0] and name_split[-1] == name_2_split[-1]:
                return True
    return False

def new_title(title):
    t_split = title.split()
    new_title = ""
    current = ""
    n_chars = 10
    for word in t_split:
        current += " "+word
        if len(current) > n_chars:
            new_title += current
            new_title += "\n"
            current = ""
    new_title += current
    return new_title

def node_size(n):
    if n:
        return 5000
    else:
        return 1500

def font_size(n):
    if n:
        return 5
    else:
        return 3

def generate_book_graph_figure(chapter: int, book: str, sections: set[tuple[int, int, int]]):
    book_df = pd.read_pickle("book_embeddings.pkl")
    df = book_df[(book_df['file_type'].isin([book])) & (book_df['0'].isin([chapter]))]
    t = df['title'].tolist()
    s = df['sections'].tolist()

    titles = [s[i] +": "+ t[i] for i in range(len(df))]
    first = df['1'].tolist()
    second = df['2'].tolist()
    G = nx.DiGraph()
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
            node_colors.append("green")
        else:
            node_colors.append("lightblue")

    A = nx.nx_agraph.to_agraph(G)

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
    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors, ax=ax)
    for i, (n, (x, y)) in enumerate(pos.items()):
        if first[i] == 0 and second[i] == 0:
            ax.text(x, y - 7 * len(n.split("\n")), n, fontsize=font_sizes[i], ha='center', va='center')
        else:
            ax.text(x, y, n, fontsize=font_sizes[i], ha='center', va='center')

    ax.set_axis_off()
    return fig


"""
Nodes
"""

def InformationNode(state: State) -> State:
    query = state.get("query")

    query_description = state.get("query_description")

    if query_description == None:

        prompt = [{"role": "system", "content": f"""
        You are an assistant for the Matlab Reservoir Toolbox developed by SINTEF.
        You are going to extract keywords, authors and a problem description from the user query.
        The keywords should help distinguish different SINTEF researchers MRST expertize fields,
        so you should be very specific when generating keywords. For example, do NOT include keywords like
        'reservoir simulation' or 'numerical simulation'.
                   
        You can use the tool "web_search_mrst":
        name: {web_search_mrst.name}
        description: {web_search_mrst.description}

        to get content and further links from the mrst webpage, if you feel like that will help you generate keywords.
        Here's a list of possible inputs, if you are going to use the tool, choose the most relevant one:

        {web_search_mrst.invoke(input = 'https://www.sintef.no/projectweb/mrst/modules/')[1]}

        """},
        {"role": "user", "content": query}]

        query_description = strong_client.with_structured_output(QueryDescription).invoke(prompt)

        return {"query_description": query_description}

    else:
        tools_calls = state.get('tools_calls')
        prompt = [{"role": "system", "content": f"""
        You are an assistant for the Matlab Reservoir Toolbox developed by SINTEF.
        You are going to extract keywords, authors and a problem description from the user query.
        The keywords should help distinguish different SINTEF researchers MRST expertize fields,
        so you should be very specific when generating keywords. For example, do NOT include keywords like
        'reservoir simulation' or 'numerical simulation'.
                   
        You can use the tool "web_search_mrst":
        name: {web_search_mrst.name}
        description: {web_search_mrst.description}

        You have already generated keywords, and called the tool once.
        Input should be one of the links found in your latest tool call.
        Here is your latest answer, and here is your latest tool call.

        query_description:
        {query_description}

        tools_calls:
        {tools_calls}

        """},
        {"role": "user", "content": query}]

        response = strong_client.with_structured_output(QueryDescription).invoke(prompt)
        response.authors = query_description.authors
        response.keywords = response.keywords + query_description.keywords

        return {"query_description": response}

def ToolsNode(state: State) -> State:
    n_tool_calls = state.get('n_tool_calls') if state.get('n_tool_calls') != None else 0
    link = state.get('query_description').tools_input
    return {"tools_calls": [web_search_mrst.invoke(input = link)], "visited_links": [link], 'n_tool_calls': n_tool_calls+1}

def RetrieveNode(state: State) -> State:
    df = pd.read_pickle('book_embeddings.pkl')
    df = df[df['file_type'].isin(['Advanced Book'])]
    query_description = state.get('query_description')
    vector = vector_embedding_model.encode(query_description.keywords + [query_description.problem_description])
    embeddings = np.array(df['embedding'].tolist())
    dot_prod = np.einsum('ij,kj->ki', vector, embeddings)
    vec_prod = np.einsum('i,k->ki',np.linalg.norm(vector, axis = -1),np.linalg.norm(embeddings, axis = -1))
    cosines = dot_prod/vec_prod
    print(f"Max cosine found:  {np.max(cosines)} \n")
    df['cosine'] = np.max(cosines, axis = -1)
    df = df[df['cosine'] >= 0.6]

    return {"df": df}

def GenerateBookNode(state: State) -> State:
    query = state.get('query')
    df = state.get("df")

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

    figures = []

    for c, b in zipped_book:
        figures.append(generate_book_graph_figure(c, b, sections=s))

    df = df.sort_values(by = 'cosine', ascending=False).head(5)

    context = "\n\n".join(["Section " + sections[i] + " in book: " + book[i] + "\n Title" + titles[i] + "\n Authors: " + ",\t".join(authors[i]) + "\n Content: " + content[i] for i in range(len(df))])
    msg = [{"role": "system", "content": f"""You are the Generator in a RAG application.
            You are going to state in which book and which section the user can learn more,
            and who they should contact based on the authors of the relevant book sections.
            Do not mention the title.
            \n Context:\n {context}"""}, {"role": "user", "content": query}]
    return {"response": weak_client.invoke(msg).content, "figures": figures}

def RetrieveAuthorNode(state: State) -> State:
    df = pd.read_pickle('book_embeddings.pkl')
    authors_names = state.get('query_description').authors
    df = df[df['authors'].apply(lambda x: bool([True for a in x if name_in(a, authors_names)]))]
    return {"df": df}

def GenerateAuthorNode(state: State) -> State:
    df = state.get("df")
    query = state.get("query")
    titles = df['title'].tolist()
    authors = [", ".join(a) for a in df['authors']]
    context = "\n\n".join([" title: " + i+"\n Authors: "+j for i,j in zip(titles, authors)])
    msg = [{"role": "system", "content": "You are the Generator in RAG application. "+
        "You are going to answer the users query, based on the context given. If there are multiple researchers mentioned in the users query, "+
        "answer the query seperately for each researcher. "+
        "The context given is titles of retrieved documents, and their respective authors. "+
        "\n Context:\n" + context}, {"role": "user", "content": query}]
    return {"response": weak_client.invoke(msg).content}

def SearchNode(state: State) -> State:
    return {}

"""
Routers
"""

def RetrievalRouter(state: State) -> Literal["ToolsNode","RetrieveNode","RetrieveAuthorNode"]:
    queryDescription = state.get('query_description')
    if queryDescription.tools:
        if state.get('visited_links') == None:
            return "ToolsNode"
        if state.get('n_tool_calls') == None:
            return "ToolsNode"
        if queryDescription.tools_input not in state.get('visited_links') and state.get('n_tool_calls')<3:
            return "ToolsNode"
    
    if queryDescription.authors:
        return "RetrieveAuthorNode"
    else:
        return "RetrieveNode"

def BookRouter(state: State) -> Literal["GenerateBookNode", "SearchNode"]:
    if len(state.get('df')):
        return "GenerateBookNode"
    else:
        return "SearchNode"

"""
Setting up the graph
"""

graph_builder = StateGraph(State)

graph_builder.add_node("InformationNode", InformationNode)
graph_builder.add_node("ToolsNode", ToolsNode)
graph_builder.add_node("RetrieveNode", RetrieveNode)
graph_builder.add_node("GenerateBookNode", GenerateBookNode)
graph_builder.add_node("RetrieveAuthorNode", RetrieveAuthorNode)
graph_builder.add_node("GenerateAuthorNode", GenerateAuthorNode)
graph_builder.add_node("SearchNode", SearchNode)

graph_builder.add_edge(START, "InformationNode")
graph_builder.add_conditional_edges("InformationNode", RetrievalRouter, {"ToolsNode": "ToolsNode", "RetrieveNode": "RetrieveNode", "RetrieveAuthorNode": "RetrieveAuthorNode"})
graph_builder.add_conditional_edges("RetrieveNode", BookRouter, {"GenerateBookNode":"GenerateBookNode", "SearchNode":"SearchNode"})
graph_builder.add_edge("ToolsNode", "InformationNode")
graph_builder.add_edge("RetrieveAuthorNode", "GenerateAuthorNode")
graph_builder.add_edge("GenerateBookNode", "SearchNode")
graph_builder.add_edge("SearchNode", END)
graph_builder.add_edge("GenerateAuthorNode", END)

graph = graph_builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path='graph_vizualization.png')