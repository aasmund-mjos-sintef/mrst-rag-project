from sentence_transformers import SentenceTransformer
import pandas as pd

vector_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def embed_again():
    df = pd.read_pickle('book_embeddings.pkl')
    embeddings = []
    titles = df['title'].tolist()
    content = df['content'].tolist()
    for i in range(len(df)):
        print(f"Embedding file nr. {i+1} out of {len(df)}", end = "\r")
        text = titles[i] + content[i]
        embeddings.append(vector_embedding_model.encode(text))
    
    df['embedding'] = embeddings
    df.to_pickle('book_embeddings.pkl')
    print("\n")
    print("Sucessfully embedded!")

embed_again()