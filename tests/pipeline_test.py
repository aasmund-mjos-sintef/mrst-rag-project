import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from graph import *

while True:
    question = input("Enter question: ")
    print("\n")
    state = graph.invoke(State(query = question))
    print(state.get('response'))
    figures = state.get('figures')
    if figures != None:
        n = len(figures)
        for i,fig in enumerate(figures):
            ready = input(f"Press enter for chapter image nr. {i+1} out of {n}: ")
            fig.savefig("results/chapter-im.svg", format="svg", bbox_inches="tight")
    print("\n")