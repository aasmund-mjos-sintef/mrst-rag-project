from sentence_transformers import SentenceTransformer
import pandas as pd

vector_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def create_embedding():
    introduction_df = pd.read_pickle('introduction_book_df.pkl')
    advanced_df = pd.read_pickle('advanced_book_df.pkl')
    df = pd.concat([introduction_df, advanced_df])
    embeddings = []
    titles = df['section_name'].tolist()
    content = df['content'].tolist()
    for i in range(len(df)):
        print(f"Embedding file nr. {i} out of {len(df)}", end = "\r")
        text = titles[i] + content[i]
        embeddings.append(vector_embedding_model.encode(text))
    
    df['embedding'] = embeddings
    df.to_pickle('test_embeddings.pkl')
    print("Sucessfully embedded!")

create_embedding()