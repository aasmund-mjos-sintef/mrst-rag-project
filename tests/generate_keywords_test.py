def test_keywords():
    from mrst_competence_query.graph import State, InformationNode
    from mrst_competence_query.questions import example_questions
    
    keywords_from_question_before_filtering = {}
    keywords_from_question_after_filtering = {}
    for q in example_questions:
        state = InformationNode(State(query = q))
        keywords_from_question_after_filtering[q] = state.get('query_description').keywords
        keywords_from_question_before_filtering[q] = state.get('keywords_before_filtering')
    
    return keywords_from_question_before_filtering, keywords_from_question_after_filtering

def print_keywords():
    b_f, a_f  = test_keywords()
    questions = b_f.keys()
    for q in questions:
        print('-'*40 + '\n' + q + '\n')
        for k in a_f.get(q):
            print(k)
        print('')

def one_question_keywords_table(questions):
    string = f''''''
    for q in questions[:-1]:
        string += q + r'\newline'
        string += f"""
"""
    string += questions[-1]
    return string

def print_keywords_to_latex_table():
    b_f, a_f = test_keywords()
    q = list(a_f.keys())
    print(rf'''
\scriptsize
\begin{{center}}
\captionsetup{{hypcap=false}}
\captionof{{table}}{{Before filtering}} \label{{tab:keywords_before_filtering}}
\begin{{tabular}}{{|m{{5cm}}|m{{5cm}}|m{{5cm}}|}}
\hline
\textbf{{{q[0]}}} &
\textbf{{{q[1]}}} &
\textbf{{{q[2]}}} \\
\hline
{one_question_keywords_table(b_f.get(q[0]))}& 
{one_question_keywords_table(b_f.get(q[1]))}&
{one_question_keywords_table(b_f.get(q[2]))}\\
\hline
\textbf{{{q[3]}}} &
\textbf{{{q[4]}}} &
\textbf{{{q[5]}}} \\
\hline
{one_question_keywords_table(b_f.get(q[3]))}& 
{one_question_keywords_table(b_f.get(q[4]))}&
{one_question_keywords_table(b_f.get(q[5]))}\\
\hline
\end{{tabular}}
\end{{center}}
\normalsize

\scriptsize
\begin{{center}}
\captionsetup{{hypcap=false}}
\captionof{{table}}{{After filtering}} \label{{tab:keywords_after_filtering}}
\begin{{tabular}}{{|m{{5cm}}|m{{5cm}}|m{{5cm}}|}}
\hline
\textbf{{{q[0]}}} &
\textbf{{{q[1]}}} &
\textbf{{{q[2]}}} \\
\hline
{one_question_keywords_table(a_f.get(q[0]))}& 
{one_question_keywords_table(a_f.get(q[1]))}&
{one_question_keywords_table(a_f.get(q[2]))}\\
\hline
\textbf{{{q[3]}}} &
\textbf{{{q[4]}}} &
\textbf{{{q[5]}}} \\
\hline
{one_question_keywords_table(a_f.get(q[3]))}& 
{one_question_keywords_table(a_f.get(q[4]))}&
{one_question_keywords_table(a_f.get(q[5]))}\\
\hline
\end{{tabular}}
\end{{center}}
\normalsize

''')

print_keywords_to_latex_table()