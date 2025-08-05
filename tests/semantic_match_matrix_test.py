def test_semantic_match_matrices(): 
    from mrst_competence_query.graph import State, vector_embedding_model, InformationNode, load_dataframe, get_bigram_freq
    from mrst_competence_query.questions import example_questions
    import numpy as np
    return_string = rf''''''
    for q in example_questions[:2]:

        keywords = InformationNode(State(query = q)).get('query_description').keywords
        df = load_dataframe('mrst_abstracts_embedding.pkl')
        keywords_embedded = vector_embedding_model.encode(keywords)

        embeddings = np.array(df['embedding'].tolist())
        dot_prod = np.einsum('ij,kj->ki', keywords_embedded, embeddings)
        vec_prod = np.einsum('i,k->ki',np.linalg.norm(keywords_embedded, axis = -1),np.linalg.norm(embeddings, axis = -1))
        cosines = dot_prod/vec_prod
        df['cosine'] = np.max(cosines, axis = -1)
        threshold = min([0.5, np.max(cosines)-0.2])

        sorted_df = df[df['cosine'] > threshold].sort_values(by = 'cosine').head(20)
        # sorted_df = df[df['cosine'] > threshold]
        n_bigrams = 5

        top_bigrams = [[tup[0] for tup in get_bigram_freq(x).most_common(n_bigrams)] for x in sorted_df['content'].tolist()]

        top_bigrams_embedded = np.array([vector_embedding_model.encode(x) for x in top_bigrams])

        dot_prod = np.einsum('xnj,kj->xnk', top_bigrams_embedded, keywords_embedded)
        vec_prod = np.einsum('xn,k->xnk',np.linalg.norm(top_bigrams_embedded, axis = -1),np.linalg.norm(keywords_embedded, axis = -1))
        cosines = dot_prod/vec_prod

        different_methods = ['Average Query-Keyword Score', 'Average Article-Keyword Score', 'Total Average']

        return_string += rf'''

\scriptsize
\begin{{center}}
\captionsetup{{hypcap=false}}
\captionof{{table}}{{Articles retrieved for question: {q}}} \label{{tab:{q}}}
\begin{{tabular}}{{|m{{5cm}}|m{{5cm}}|m{{5cm}}|}}
\hline
{' &'.join(different_methods)}\\
\hline
'''
        paper_strings = []
        for method in ['Average Query-Keyword Score', 'Average Article-Keyword Score', 'Total Average']:
            if method == 'Average Query-Keyword Score':
                sorted_df['avg_high_cosine'] = np.mean(np.max(cosines, axis = 2), axis = -1)
            elif method == 'Average Article-Keyword Score':
                sorted_df['avg_high_cosine'] = np.mean(np.max(cosines, axis = 1), axis = -1)
            else:
                sorted_df['avg_high_cosine'] = np.mean(cosines, axis=(1,2))
            top_papers = sorted_df.sort_values(by = 'avg_high_cosine', ascending=False).head(5)
            paper_strings.append( rf"""\newline
""".join([rf"- {t}" for t in top_papers['titles']]))
        return_string += rf'''{rf""" &
""".join(paper_strings)}\\
\hline
\end{{tabular}}
\end{{center}}
\normalsize
'''
    return return_string

print(test_semantic_match_matrices())