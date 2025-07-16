from git import Repo
import os
from sentence_transformers import SentenceTransformer
import pandas as pd

embedding_model_name = 'microsoft/codebert-base'

possible_to_check_out = """
- microsoft/codebert-base
- microsoft/codebert-base-mlm
- microsoft/unixcoder-base
- microsoft/codexglue-code-search-net
- Salesforce/codet5-base
"""

new_dir = "mrst_cloned"

def get_last_commit_info(repo_path, file_path):
    try:
        if repo_path.startswith("http"):

            tempfile_version = False

            if tempfile_version:
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    print(f"Cloning repository from {repo_path} to temporary directory {temp_dir}")
                    repo = Repo.clone_from(repo_path, temp_dir)
                    repo_path = temp_dir
            else:

                if not os.path.exists(new_dir):
                    print(f"Cloning repository from {repo_path} to temporary directory {new_dir}")
                    repo = Repo.clone_from(url=repo_path, to_path=new_dir)
                else:
                    print(f"Repository already exists at {repo_path}, using existing repository.")
                repo_path = os.path.abspath(new_dir)

        repo = Repo(repo_path)
        commits = list(repo.iter_commits(paths=file_path, max_count=1))

        if commits:
            last_commit = commits[0]
            author_email = last_commit.author.email
            author_name = last_commit.author.name
            commit_message = last_commit.message.strip()
            return author_name, author_email, commit_message
        else:
            return None, None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

repository_path = "https://github.com/SINTEF-AppliedCompSci/MRST"
target_file = "README.md"

# author, email, message = get_last_commit_info(repository_path, target_file)

# if author and message:
#     print(f"Last committed by: {author}")
#     print(f"Author email: {email}")
#     print(f"Commit message: {message}")
# else:
#     print(f"Could not find commit information for {target_file}")

def get_commit_frequency_numbers(repo_path, file_path):
    repo = Repo(repo_path)
    commits = list(repo.iter_commits(paths=file_path))

    if commits:
        freq_dict_name = {}
        freq_dict_email = {}
        for c in commits:
            author_email = c.author.email
            author_name = c.author.name
            co_authors = c.co_authors
            freq_dict_email[author_email] = freq_dict_email.get(author_email, 0) + 1
            freq_dict_name[author_name] = freq_dict_name.get(author_name, 0) +1
            for c_a in co_authors:
                author_email = c_a.email
                author_name = c_a.name
                freq_dict_email[author_email] = freq_dict_email.get(author_email, 0) + 1
                freq_dict_name[author_name] = freq_dict_name.get(author_name, 0) +1

        return freq_dict_email, freq_dict_name
    else:
        return None, None

def embed_github_repository(file_condition: callable, text_splitter: str = None):

    vector_embedding_model = SentenceTransformer(embedding_model_name)
    print("Loading documents from Git repository...")

    repo = Repo(path = new_dir)

    file_paths = []
    metadatas = []
    contents = []
    embeddings = []
    authors = []
    emails = []
    messages = []

    total = 0

    for item in repo.tree().traverse():
        file_path = item.path
        if file_condition(file_path):
            total += 1

    i = 0
    for item in repo.tree().traverse():
        file_path = item.path
        if file_condition(file_path):
            i+=1
            print(f"Embedding file {i} out of {total}", end = "\r")
            with open(new_dir + "/" +  file_path, 'rb') as f:

                content = f.read()
                file_type = os.path.splitext(item.name)[1]
                metadata = {
                    "source": file_path,
                    "file_path": file_path,
                    "file_name": item.name,
                    "file_type": file_type,
                }

                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError as e:
                    text_content = str(content)

            author, email, message = get_last_commit_info(new_dir, file_path)

            if text_splitter == None:
                embedding = vector_embedding_model.encode(text_content)

                file_paths.append(file_path)
                metadatas.append(metadata)
                contents.append(text_content)
                embeddings.append(embedding)
                authors.append(author)
                emails.append(email)
                messages.append(message)
            else:
                content_chunks = text_content.split(text_splitter)
                for j in range(1,len(content_chunks)):
                    content_chunks[j] = text_splitter + content_chunks[j]
                
                for c in content_chunks:
                    embedding = vector_embedding_model.encode(c)

                    file_paths.append(file_path)
                    metadatas.append(metadata)
                    contents.append(c)
                    embeddings.append(embedding)
                    authors.append(author)
                    emails.append(email)
                    messages.append(message)
    
    df = pd.DataFrame({"file_path": file_paths,
                       "content": contents,
                       "author": authors,
                       "email": emails,
                       "message": messages,
                       "embedding": embeddings,
                       "metadata": metadatas})
    
    df.to_pickle("mrst_repository_embeddings.pkl")

# embed_github_repository(file_condition = lambda x: x[-2:]==".m" or "README" in x)
# email, name = get_commit_frequency_numbers(repo_path = new_dir, file_path = "core")
# print(email)
# print(name)