from git import Repo
import os
from sentence_transformers import SentenceTransformer
import pandas as pd

from dotenv import load_dotenv
import os
load_dotenv()
cloned_repo_path = os.getenv("MRST_REPOSITORY_PATH")

code_embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'


extract_string = """Copyright 2009-2024 SINTEF Digital, Mathematics & Cybernetics.

This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).

MRST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MRST is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MRST.  If not, see <http://www.gnu.org/licenses/>."""

class GitAgent:
    def __init__(self, repo_path = cloned_repo_path):
        if repo_path.startswith("http"):

            tempfile_version = False

            if tempfile_version:
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    print(f"Cloning repository from {repo_path} to temporary directory {temp_dir}")
                    self.repo = Repo.clone_from(repo_path, temp_dir)
            else:

                if not os.path.exists(cloned_repo_path):
                    print(f"Cloning repository from {repo_path} to temporary directory {cloned_repo_path}")
                    self.repo = Repo.clone_from(url=repo_path, to_path=cloned_repo_path)
                else:
                    print(f"Repository already exists at {repo_path}, using existing repository.")
                    repo_path = os.path.abspath(cloned_repo_path)
                    self.repo = Repo(repo_path)
        else:
            self.repo = Repo(repo_path)

    def get_last_commit_info(self, file_path):
        try:
            repo = self.repo
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

    def get_commit_frequency_numbers(self, file_path):
        repo = self.repo
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

    def embed_github_repository_general(self, file_condition: callable, text_splitter: str = None):

        vector_embedding_model = SentenceTransformer(code_embedding_model_name)
        print("Loading documents from Git repository...")
        repo = self.repo

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
                with open(cloned_repo_path + "/" +  file_path, 'rb') as f:

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

                author, email, message = self.get_last_commit_info(file_path)

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

    def embed_github_repository_content_README(self):

        vector_embedding_model = SentenceTransformer(code_embedding_model_name)
        print("Loading documents from Git repository...")
        repo = self.repo

        file_paths = []
        metadatas = []
        contents = []
        embeddings = []
        authors = []
        emails = []
        messages = []

        total = 0

        file_condition = lambda x: "contents.m" in x.lower() or "README" in x

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
                with open(cloned_repo_path + "/" +  file_path, 'rb') as f:

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

                if "contents.m" in file_path.lower():
                    index = text_content.find(extract_string)
                    text_content = text_content[:index] + text_content[1+index + len(extract_string):]

                author, email, message = self.get_last_commit_info(file_path)

                embedding = vector_embedding_model.encode(text_content)

                file_paths.append(file_path)
                metadatas.append(metadata)
                contents.append(text_content)
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