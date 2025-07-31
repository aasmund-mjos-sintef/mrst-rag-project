import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

def fetch_abstracts(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {
        "Accept": "application/json"
    }
    params = {
        "query": query,
        "fields": "title,abstract,year,url",
        "limit": 1,
        "offset": 0
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    print(data)

start_links = ["https://www.sintef.no/projectweb/mrst/publications/papers-by-mrst-team/",
               "https://www.sintef.no/projectweb/mrst/publications/proceedings-by-mrst-team/",
               "https://www.sintef.no/projectweb/mrst/publications/papers-by-others/",
               "https://www.sintef.no/projectweb/mrst/publications/proceedings-papers/",
               "https://www.sintef.no/projectweb/mrst/publications/papers-by-others2/",
               "https://www.sintef.no/projectweb/mrst/publications/phd-theses/",
               "https://www.sintef.no/projectweb/mrst/publications/master-theses/"]

authors = {}

links = []
titles = []
paper_authors = []

i = 0

for mrst_link in start_links:
    response = requests.get(mrst_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    for section in soup.find_all('div', class_ = "rich-text-field"):
            for article in section.find_all("li"):
                    text_snippet = article.get_text()
                    if i < 10:
                        fetch_abstracts(text_snippet)
                    i += 1
                    