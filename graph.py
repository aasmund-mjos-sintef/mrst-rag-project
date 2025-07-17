from langchain_openai import ChatOpenAI
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
from langgraph.prebuilt import create_react_agent
from collections import Counter
from classes.GitAgent import GitAgent
import warnings
warnings.filterwarnings('ignore')

import re
from nltk.corpus import stopwords
from nltk import download as nltk_download
nltk_download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_openai_api_key = os.getenv("LANGCHAIN_OPENAI_API_KEY")

tools = [web_search_mrst]

strong_client = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=langchain_openai_api_key)
beast_client = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=langchain_openai_api_key)
tools_agent = create_react_agent(strong_client, tools)
weak_client = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key = langchain_openai_api_key)

vector_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

"""
Helper classes -- Called with client.with_structured_output(ClassName)
"""

class QueryDescriptionWithTools(BaseModel):
    keywords: List[str] = Field(description = "Keywords related to the users query")
    authors: List[str] = Field(description = "The authors mentioned in the users query")
    problem_description: str = Field(description = "The users problem compacted into one sentence")
    tools: bool = Field(description = "Wether or not should call tools")
    tools_input: str = Field(description = "The input to the tool")

class QueryDescription(BaseModel):
    keywords: List[str] = Field(description = "Keywords related to the users query")
    authors: List[str] = Field(description = "The authors mentioned in the users query")
    problem_description: str = Field(description = "The users problem compacted into one sentence")

class CodingKeyWords(BaseModel):
    keywords: List[str] = Field(description = "Key code object names or code words/functions etc based on the provided code.")

class State(TypedDict):
    query: str
    code_query: str
    coding_keywords: List[str]
    query_description : QueryDescriptionWithTools
    attempts: int
    df: pd.DataFrame
    code_df: pd.DataFrame
    book_response: str
    figures: List[Figure]
    tools_calls: List[tuple[str, List[str]]]
    visited_link: str
    chapter_info: tuple[int, str]
    authors_papers_dict: dict[str, int]
    authors_chunks_dict: dict[str, int]
    authors_total_papers_dict: dict[str, int]
    authors_total_chunks_dict: dict[str, int]
    authors_papers_percentage_dict: dict[str, float]
    authors_chunks_percentage_dict: dict[str, float]
    authors_relevance_score: dict[str, float]
    github_authors_relevance_score: dict[str, float]
    sources: set[str]
    cosine_dict: dict[str, tuple[float, float]]

class SpecificScore(BaseModel):
    specific_scores: List[int] = Field(description = "List of specific_score is a number from 1 to 10 defining how specific or general the keywords are related to the scientific field of reservoir simulation. 10 is extremely specific, and 1 is extremely general")

class Authors(BaseModel):
    authors: List[str] = Field(description = "List of the authors mentioned in the users query. Has to be specifically mentioned by the user!")

"""
Helper functions
"""

def split_by_more(string: str, chars: List[str] = []):
    if chars == []:
        return string.split()
    else:
        for method, next_method in zip(chars, chars[1:]):
            string = next_method.join(string.split(method))
        return string.split(chars[-1])

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
    
def get_bigram_freq(text):
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\w+', text.lower()) 

    # Remove stop words and words with less than 3 characters
    words = [word for word in words if word not in stop_words and len(word) > 2]

    # Create list of bigrams
    bigrams = zip(words, words[1:]) # pair each word (words) with the next word (words[1:])
    bigram_list = [' '.join(bigram) for bigram in bigrams] # join the words in the bigram with a space

    # Count the frequency of each bigram
    return Counter(bigram_list)

def get_top_bigrams(df):
    total_text = ""
    for c in df['content'].tolist():
        total_text += c

    counter = get_bigram_freq(total_text)
    return counter.most_common(10)

def generate_book_graph_figure(chapter: int, book: str, sections: set[tuple[int, int, int]]):
    book_df = pd.read_pickle("datasets/book_embeddings.pkl")
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
            ax.text(x, y, n, fontsize=font_sizes[i], ha='center', va='center')
        else:
            ax.text(x, y, n, fontsize=font_sizes[i], ha='center', va='center')

    max_radius = max(np.sqrt(s / 10) for s in node_sizes)
    x_vals, y_vals = zip(*pos.values())

    ax.set_xlim(min(x_vals) - 3*max_radius, max(x_vals) + 3*max_radius)
    ax.set_ylim(min(y_vals) - 3*max_radius, max(y_vals) + 7*max_radius)
    
    ax.set_axis_off()
    return fig


"""
Nodes
"""

def InformationNode(state: State) -> State:
    query = state.get("query")

    conventional = True # choose if normal method to call tools

    if conventional:

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
        Illegal and bad keywords are:
            - reservoir simulation
            - numerical mathematics
        and other general queries that has something to do with MRST in general, but not specifically chemical eor.
                
        Format the output as such:
        {{
            keywords: [<list_of_keywords>],
            problem_description: <problem_description>
        }}
                
        You can use the tool 'web_search_mrst' tp get content and further links from the mrst webpage.
        Only use the tool if you're sure the link is relevant.
        Here's a list of possible start links, if you are going to use the tool, choose the most relevant one:

            {web_search_mrst.invoke(input = 'https://www.sintef.no/projectweb/mrst/modules/')[1]}

        """),
        ("user", query)]

        response = tools_agent.invoke({"messages": msg})
        answer = response['messages'][-1].content

        k = 'keywords: '
        p = 'problem_description: '

        keywords_index = answer.find(k)
        problem_index = answer.find(p)

        keywords_split = answer[keywords_index + len(k):problem_index]
        problem_split = answer[problem_index + len(p):]

        keywords = [i for i in split_by_more(keywords_split, ['[',']','"',',']) if i.strip()][:-1]
        problem_description = [i for i in split_by_more(problem_split, ['"']) if i.strip()][0]

        authors = strong_client.with_structured_output(Authors).invoke([{"role": "system", "content": "You are going to extract SINTEF researchers from a users query"}, {"role": "user", "content": query}]).authors

        authors = [a for a in authors if "sintef" not in a.lower()]

        if len(query) < 500:    # Since we use an expensive model, we only allow smaller prompts into the llm

            keywords_string = ""
            for k in keywords:
                keywords_string += "-"+k
                keywords_string += "\n"

            n = len(keywords)

            sys_msg = f"""You are going to determine how specific these keywords are related to the field of reservoir simulation in relation to the users query.
            You are NOT going to determine the relevance to the users query, but you are going to determine how specific it is related to reservoir simulation and the query.
            For example, keywords like 'reservoir simulation', 'numerical mathematics' should always have a low score like 1 or 2. 
            For example, keywords like 'chemical enhanced oil recovery' should be very high if the query is 'What can you tell me about chemical eor'
            Generate score for all keywords, so a total of {n}.
            Query:\n{query}
            
            Keywords:\n {keywords_string}"""

            specific_score = []

            i = 0
            while len(specific_score) != n and i<3:
                i+=1
                specific_score = beast_client.with_structured_output(SpecificScore).invoke([{"role": "system", "content": sys_msg}]).specific_scores

            if i < 3:
                keywords = [keywords[j] for j in range(n) if specific_score[j]>6]

        query_description = QueryDescriptionWithTools(keywords=keywords, problem_description=problem_description, authors=authors, tools=False, tools_input="")
        return {"query_description": query_description}

    else:
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

            query_description = strong_client.with_structured_output(QueryDescriptionWithTools).invoke(prompt)

            return {"query_description": query_description}

        else:
            tools_calls = state.get('tools_calls')

            prompt = [{"role": "system", "content": f"""
            You are an assistant for the Matlab Reservoir Toolbox developed by SINTEF.
            You are going to extract keywords, authors and a problem description from the user query.
            The keywords should help distinguish different SINTEF researchers MRST expertize fields,
            so you should be very specific when generating keywords. For example, do NOT include keywords like
            'reservoir simulation' or 'numerical simulation'.
                    
            You have used the tool "web_search_mrst":
            name: {web_search_mrst.name}
            description: {web_search_mrst.description}

            Here is your tool call.

            tools_calls:
            {tools_calls[0]}

            """},
            {"role": "user", "content": query}]

            response = strong_client.with_structured_output(QueryDescription).invoke(prompt)
            return_val = QueryDescriptionWithTools(
                authors = query_description.authors,
                keywords = list(set(response.keywords + query_description.keywords)),
                problem_description = response.problem_description,
                tools = False,
                tools_input = ""
                )

            return {"query_description": return_val}

def SearchMRSTModulesNode(state: State) -> State:
    link = state.get('query_description').tools_input
    return {"tools_calls": [web_search_mrst.invoke(input = link)], "visited_link": link}

def RetrieveNode(state: State) -> State:
    df = pd.read_pickle('datasets/book_embeddings.pkl')
    df = df[df['file_type'].isin(['Advanced Book'])]
    query_description = state.get('query_description')
    vector = vector_embedding_model.encode(query_description.keywords + [query_description.problem_description])
    embeddings = np.array(df['embedding'].tolist())
    dot_prod = np.einsum('ij,kj->ki', vector, embeddings)
    vec_prod = np.einsum('i,k->ki',np.linalg.norm(vector, axis = -1),np.linalg.norm(embeddings, axis = -1))
    cosines = dot_prod/vec_prod
    print(f"Max cosine found:  {np.max(cosines)}")
    df['cosine'] = np.max(cosines, axis = -1)
    df = df[df['cosine'] >= 0.65]

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
    return {"book_response": weak_client.invoke(msg).content, "figures": figures, "chapter_info": list(zipped_book)}

def RetrieveAuthorNode(state: State) -> State:
    df = pd.read_pickle('datasets/book_embeddings.pkl')
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
    return {"book_response": weak_client.invoke(msg).content}

def SearchNode(state: State) -> State:
    df = pd.read_pickle('datasets/folk_ntnu_embeddings.pkl')
    query_description = state.get('query_description')
    vector = vector_embedding_model.encode(query_description.keywords + [query_description.problem_description])
    embeddings = np.array(df['embedding'].tolist())
    dot_prod = np.einsum('ij,kj->ki', vector, embeddings)
    vec_prod = np.einsum('i,k->ki',np.linalg.norm(vector, axis = -1),np.linalg.norm(embeddings, axis = -1))
    cosines = dot_prod/vec_prod
    df['cosine'] = np.max(cosines, axis = -1)
    threshold = 0.55
    max_cosine = np.max(cosines)

    print("Threshold set to: ", threshold)

    sorted_df = df[df['cosine'] > threshold]
    print(f"Found {len(sorted_df)} chunks.")
    sources = set(sorted_df['source'].tolist())
    print(f"Found {len(sources)} papers.")

    authors_papers_dict = {} # Total number of papers retrieved per author
    authors_chunks_dict = {} # Total number of chunks retrieved per author
    authors_total_papers_dict = {} # Total number of papers per retrieved author
    authors_total_chunks_dict = {} # Total number of chunks per retrieved author
    authors_papers_percentage_dict = {}
    authors_chunks_percentage_dict = {}

    authors = set()

    for source in sources:
        source_df = sorted_df[sorted_df['source'] == source]
        authors_from_source = source_df['authors'].tolist()[0]
        for a in authors_from_source:
            authors.add(a)

            cosine = source_df['cosine'].tolist()

            for i in range(len(source_df)):
                authors_chunks_dict[a] = authors_chunks_dict.get(a, 0) + 1 + 1*(cosine[i]-threshold)/(max_cosine-threshold)

            authors_papers_dict[a] = authors_papers_dict.get(a, 0) + 1
        
    for a in authors:
        authors_df = df[df['authors'].apply(lambda x: a in x) == True]
        authors_total_chunks_dict[a] = len(authors_df)
        authors_papers_df = authors_df[authors_df['chunk'] == 0]
        authors_total_papers_dict[a] = len(authors_papers_df)

    for a in authors:
        authors_papers_percentage_dict[a] = authors_papers_dict[a]/authors_total_papers_dict[a]
        authors_chunks_percentage_dict[a] = 0.5*authors_chunks_dict[a]/authors_total_chunks_dict[a]

    max_papers_per_person = max(authors_total_papers_dict.values()) if authors else 0
    max_chunks_per_person = max(authors_total_chunks_dict.values()) if authors else 0

    authors_relevance_score = {}
    gamma = 0.2

    for a in authors:
        authors_relevance_score[a] = 100*authors_chunks_percentage_dict[a]*authors_papers_percentage_dict[a]*(gamma*max_papers_per_person + authors_total_papers_dict[a])/((1+gamma)*max_papers_per_person)#*(gamma*max_chunks_per_person + authors_total_chunks_dict[a])/((1+gamma)*max_chunks_per_person)

    return {"authors_chunks_dict": authors_chunks_dict,
            "authors_papers_dict": authors_papers_dict,
            "authors_total_chunks_dict": authors_total_chunks_dict,
            "authors_total_papers_dict": authors_total_papers_dict,
            "authors_chunks_percentage_dict": authors_chunks_percentage_dict,
            "authors_papers_percentage_dict": authors_papers_percentage_dict,
            "authors_relevance_score": authors_relevance_score,
            "sources": sources}

def EvaluateNode(state: State) -> State:
    df = pd.read_pickle('datasets/folk_ntnu_embeddings.pkl')
    sources = state.get('sources')
    keywords = state.get('query_description').keywords
    keywords = keywords if len(keywords) else [state.get('query_description').problem_description]
    keyword_embeddings = vector_embedding_model.encode(keywords)

    cosine_dict = {}

    for source in sources:
        top_bigrams_and_freq = get_top_bigrams(df[df['source'].isin([source])])
        top_bigrams = [bigram for bigram, freq in top_bigrams_and_freq]
        bigrams_embeddings = vector_embedding_model.encode(top_bigrams)

        # bigrams_embeddings has shape (n_bigrams , 768)
        # keyword_embeddings has shape (k_keywords, 768)

        dot_prod = np.einsum('nj,kj->nk', bigrams_embeddings, keyword_embeddings)
        vec_prod = np.einsum('n,k->nk',np.linalg.norm(bigrams_embeddings, axis = -1),np.linalg.norm(keyword_embeddings, axis = -1))
        cosines = dot_prod/vec_prod
        new_shape = (len(keywords)*len(top_bigrams),)
        sorted_cosines = np.reshape(cosines, new_shape)
        sorted_cosines.sort()
        n = new_shape[0]//4
        highest_cosines = np.zeros((n,), dtype = float)
        for i in range(0,n):
            highest_cosines[i] = sorted_cosines[new_shape[0]-i-1]

        avg_high_cosine = np.mean(highest_cosines)
        avg_cosine = np.mean(cosines)
        cosine_dict[source] = (avg_cosine, avg_high_cosine)

    return {"cosine_dict": cosine_dict}

def GitNode(state: State) -> State:
    query = state.get('query')
    code_query = state.get('code_query')
    prompt = [{"role": "system", "content": "You are going to extract a maximum of 10 code keywords related to the problem the user has based on the provided code and problem"},{"role":"user", "content": "problem:\n " + query + "\n\ncode:\n" + code_query}]
    coding_keywords = strong_client.with_structured_output(CodingKeyWords).invoke(prompt).keywords
    code_search = coding_keywords
    code_search_embeddings = np.array(vector_embedding_model.encode(code_search))
    df = pd.read_pickle("datasets/mrst_repository_embeddings.pkl")
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

    print("\n")

    return {"code_df": sorted_df, "coding_keywords": coding_keywords, "github_authors_relevance_score": total_freq_dict}

"""
Routers
"""

def RetrievalRouter(state: State) -> Literal["SearchMRSTModulesNode","RetrieveNode","RetrieveAuthorNode"]:
    queryDescription = state.get('query_description')
    if queryDescription.tools:
        return "SearchMRSTModulesNode"

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
graph_builder.add_node("SearchMRSTModulesNode", SearchMRSTModulesNode)
graph_builder.add_node("RetrieveNode", RetrieveNode)
graph_builder.add_node("GenerateBookNode", GenerateBookNode)
graph_builder.add_node("RetrieveAuthorNode", RetrieveAuthorNode)
graph_builder.add_node("GenerateAuthorNode", GenerateAuthorNode)
graph_builder.add_node("SearchNode", SearchNode)
graph_builder.add_node("EvaluateNode", EvaluateNode)
graph_builder.add_node("GitNode", GitNode)

graph_builder.add_edge(START, "InformationNode")
graph_builder.add_conditional_edges("InformationNode", RetrievalRouter, {"SearchMRSTModulesNode": "SearchMRSTModulesNode", "RetrieveNode": "RetrieveNode", "RetrieveAuthorNode": "RetrieveAuthorNode"})
graph_builder.add_conditional_edges("RetrieveNode", BookRouter, {"GenerateBookNode":"GenerateBookNode", "SearchNode":"SearchNode"})
graph_builder.add_edge("SearchMRSTModulesNode", "InformationNode")
graph_builder.add_edge("RetrieveAuthorNode", "GenerateAuthorNode")
graph_builder.add_edge("GenerateBookNode", "SearchNode")
graph_builder.add_edge("SearchNode", "EvaluateNode")
graph_builder.add_edge("EvaluateNode", "GitNode")
graph_builder.add_edge("GenerateAuthorNode", "GitNode")
graph_builder.add_edge("GitNode", END)

graph = graph_builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path='graph_vizualization.png')