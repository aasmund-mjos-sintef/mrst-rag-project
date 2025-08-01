def test_pipeline():
    from mrst_competence_query import graph

    question = "What can you tell me about chemical eor?"
    print("Starting Pipeline Test")
    graph.invoke_graph(graph.State(query = question,
                                    code_query = "",
                                    start_node = "InformationNode",
                                    clustering = False,
                                    github = False,
                                    chapter_images = False,
                                    text_answer = False))
    print("\nSUCCESS\n")

test_pipeline()