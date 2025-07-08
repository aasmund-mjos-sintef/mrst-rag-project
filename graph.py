from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List, Literal
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

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

class State(TypedDict):
    query: str
    query_description : QueryDescription
    attempts: int
    df: pd.DataFrame
    response: str

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

"""
Nodes
"""

def InformationNode(state: State) -> State:
    query = state.get("query")
    prompt = [{"role": "system", "content": """
    You are an assistant for the Matlab Reservoir Toolbox developed by SINTEF.
    You are going to extract keywords, authors and a problem description from the user query.
    The keywords should help distinguish different SINTEF researchers MRST expertize fields,
    so you should be very specific when generating keywords. For example, do NOT include keywords like
    'reservoir simulation' or 'numerical simulation'
    """},
    {"role": "user", "content": query}]

    query_description = strong_client.with_structured_output(QueryDescription).invoke(prompt)
    return {"query_description": query_description}

def RetrieveNode(state: State) -> State:
    df = pd.read_pickle('test_embeddings.pkl')
    df = df[df['file_type'].isin(['Advanced Book'])]
    query_description = state.get('query_description')
    vector = vector_embedding_model.encode(query_description.keywords + [query_description.problem_description])
    embeddings = np.array(df['embedding'].tolist())
    dot_prod = np.einsum('ij,kj->ki', vector, embeddings)
    vec_prod = np.einsum('i,k->ki',np.linalg.norm(vector, axis = -1),np.linalg.norm(embeddings, axis = -1))
    cosines = dot_prod/vec_prod
    df['cosine'] = np.max(cosines, axis = -1)
    df = df.sort_values(by = 'cosine', ascending = False)
    df = df.head(5)
    return {"df": df}

def GenerateNode(state: State) -> State:
    query = state.get('query')
    df = state.get("df")
    content = df['content'].tolist()
    authors = df['authors'].tolist()

    context = "\n\n".join(["Authors: " + ",\t".join(authors[i]) + "\n Content: " + content[i] for i in range(len(df))])
    msg = [{"role": "system", "content": "You are the Generator in RAG application."+
            "You are going to state who the user should contact about their query, "+
            "based on the author/authors behind the most relevant documents"+
            "\n Context:\n" + context}, {"role": "user", "content": query}]
    return {"response": weak_client.invoke(msg).content}

def RetrieveAuthorNode(state: State) -> State:
    df = pd.read_pickle('test_embeddings.pkl')
    authors_names = state.get('query_description').authors
    df = df[df['authors'].apply(lambda x: bool([True for a in x if name_in(a, authors_names)]))]
    return {"df": df}

def GenerateAuthorNode(state: State) -> State:
    df = state.get("df")
    query = state.get("query")
    titles = df['section_name'].tolist()
    authors = [", ".join(a) for a in df['authors']]
    context = "\n\n".join([" title: " + i+"\n Authors: "+j for i,j in zip(titles, authors)])
    msg = [{"role": "system", "content": "You are the Generator in RAG application. "+
        "You are going to answer the users query, based on the context given. If there are multiple researchers mentioned in the users query, "+
        "answer the query seperately for each researcher. "+
        "The context given is titles of retrieved documents, and their respective authors. "+
        "\n Context:\n" + context}, {"role": "user", "content": query}]
    return {"response": weak_client.invoke(msg).content}

"""
Routers
"""

def WhatKindOfRetrieval(state: State) -> Literal["RetrieveNode","RetrieveAuthorNode"]:
    if state.get('query_description').authors:
        return "RetrieveAuthorNode"
    else:
        return "RetrieveNode"


"""
Setting up the graph
"""

graph_builder = StateGraph(State)

graph_builder.add_node("InformationNode", InformationNode)
graph_builder.add_node("RetrieveNode", RetrieveNode)
graph_builder.add_node("GenerateNode", GenerateNode)
graph_builder.add_node("RetrieveAuthorNode", RetrieveAuthorNode)
graph_builder.add_node("GenerateAuthorNode", GenerateAuthorNode)

graph_builder.add_edge(START, "InformationNode")
graph_builder.add_conditional_edges("InformationNode", WhatKindOfRetrieval, {"RetrieveNode": "RetrieveNode", "RetrieveAuthorNode": "RetrieveAuthorNode"})
graph_builder.add_edge("RetrieveNode", "GenerateNode")
graph_builder.add_edge("RetrieveAuthorNode", "GenerateAuthorNode")
graph_builder.add_edge("GenerateNode", END)
graph_builder.add_edge("GenerateAuthorNode", END)

graph = graph_builder.compile()

while True:
    question = input("Enter question: ")
    print("\n")
    state = graph.invoke(State(query = question))
    print(state.get('response'))
    print("\n")
    