from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List
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
    problem_description: str = Field(description = "The users problem. ")
    code: str = Field(description = "The users code if existing")

class State(TypedDict):
    query: str
    query_description : QueryDescription
    attempts: int
    df: pd.DataFrame
    response: str

"""
Helper functions
"""

"""
Nodes
"""

def InformationNode(state: State) -> State:
    query = state.get("query")
    prompt = [{"role": "system", "content": """
    You are an assistant for the Matlab Reservoir Toolbox developed by SINTEF.
    You are going to extract a problem description from the user query. Be concise when answering
    """},
    {"role": "user", "content": query}]

    query_description = strong_client.with_structured_output(QueryDescription).invoke(prompt)
    return {"query_description": query_description}

def RetrieveNode(state: State) -> State:
    df = pd.read_pickle('test_embeddings.pkl')
    vector = vector_embedding_model.encode(state.get('query_description').problem_description)
    embeddings = np.array(df['embedding'].tolist())
    dot_prod = embeddings @ vector
    vec_prod = np.linalg.norm(vector)*np.linalg.norm(embeddings, axis = -1)
    cosines = dot_prod/vec_prod
    df['cosine'] = cosines
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

"""
Setting up the graph
"""

graph_builder = StateGraph(State)

graph_builder.add_node("InformationNode", InformationNode)
graph_builder.add_node("RetrieveNode", RetrieveNode)
graph_builder.add_node("GenerateNode", GenerateNode)

graph_builder.add_edge(START, "InformationNode")
graph_builder.add_edge("InformationNode", "RetrieveNode")
graph_builder.add_edge("RetrieveNode", "GenerateNode")
graph_builder.add_edge("GenerateNode", END)

graph = graph_builder.compile()

while True:
    question = input("Enter question: ")
    print("\n")
    state = graph.invoke(State(query = question))
    print(state.get('response'))
    print("\n")
    