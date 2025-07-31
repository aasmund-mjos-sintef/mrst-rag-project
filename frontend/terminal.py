from mrst_competence_query import graph

while True:
    question = input("Enter question: ")
    print("\n")
    state = graph.invoke_graph(graph.State(query = question,
                                            code_query = "",
                                            start_node = "InformationNode",
                                            clustering = False,
                                            github = False,
                                            chapter_images = False,
                                            text_answer = False))
    print("\n")
    print(state.get('relevance_score'))
    print("\n")