def test_chapter_images():
    from mrst_competence_query.graph import generate_book_graph_figure

    for i in range(1,15):
        fig = generate_book_graph_figure(chapter = i, book = "Advanced Book", sections=set())
        fig.savefig("images/chapter_images/" + str(i) + ".svg", format="svg", bbox_inches="tight")

test_chapter_images()