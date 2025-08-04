def test_keywords():
    from mrst_competence_query.graph import State, InformationNode
    questions =  [
        'What can you tell me about eor methods',
        'Who should I contact about coupled flow and geomechanics for CO2',
        'Who should I contact about polymer flooding',
        'Who should I call about linear solvers',
        'Who in MRST should I reach out to about Virtual Element Methods',
        'Who should I reach out to about ad-blackoil'
    ]
    keywords_from_question = {}
    for q in questions:
        keywords_from_question[q] = InformationNode(State(query = q)).get('query_description').keywords
    
    return keywords_from_question

def print_keywords():
    keywords_from_question = test_keywords()
    questions = keywords_from_question.keys()
    for q in questions:
        print('-'*40 + '\n' + q + '\n')
        for k in keywords_from_question.get(q):
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
    keywords_from_question = test_keywords()
    q = list(keywords_from_question.keys())
    print(rf'''
\scriptsize
\begin{{center}}
\captionsetup{{hypcap=false}}
\captionof{{table}}{{CAPTION HERE}} \label{{tab:}}
\begin{{tabular}}{{|m{{5cm}}|m{{5cm}}|m{{5cm}}|}}
\hline
\textbf{{{q[0]}}} &
\textbf{{{q[1]}}} &
\textbf{{{q[2]}}} \\
\hline
{one_question_keywords_table(keywords_from_question.get(q[0]))}& 
{one_question_keywords_table(keywords_from_question.get(q[1]))}&
{one_question_keywords_table(keywords_from_question.get(q[2]))}\\
\hline
\textbf{{{q[3]}}} &
\textbf{{{q[4]}}} &
\textbf{{{q[5]}}} \\
\hline
{one_question_keywords_table(keywords_from_question.get(q[3]))}& 
{one_question_keywords_table(keywords_from_question.get(q[4]))}&
{one_question_keywords_table(keywords_from_question.get(q[5]))}\\
\hline
\end{{tabular}}
\end{{center}}
\normalsize
''')

print_keywords_to_latex_table()