from sentence_transformers import SentenceTransformer
import pandas as pd

vector_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def create_embedding():
    df = pd.read_pickle('../folk_ntnu_df.pkl')
    embeddings = []
    titles = df['title'].tolist()
    content = df['content'].tolist()
    for i in range(len(df)):
        print(f"Embedding file nr. {i+1} out of {len(df)}", end = "\r")
        text = titles[i] + content[i]
        embeddings.append(vector_embedding_model.encode(text))
    
    df['embedding'] = embeddings
    df.to_pickle('folk_ntnu_embeddings.pkl')
    print("Sucessfully embedded!")

create_embedding()