import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from graph import generate_book_graph_figure

for i in range(1,15):
    fig = generate_book_graph_figure(chapter = i, book = "Advanced Book", sections=set())
    fig.savefig("chapter_images/" + str(i) + ".svg", format="svg", bbox_inches="tight")