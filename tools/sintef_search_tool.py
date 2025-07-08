from langchain_core.tools import tool
from bs4 import BeautifulSoup
import requests
from typing import List

@tool
def web_search_mrst(filename: str) -> tuple[str, List[str]]:
    """Scrape a SINTEF MRST website with filename, and return content + explicit links to other sites"""
    response = requests.get(filename)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(strip = True)
    links = ["https://sintef.no" + l['href'] for l in soup.find_all('a', href = True) if l['href'][0]=="/"]
    return (text, links)

print(web_search_mrst.run(tool_input = "https://www.sintef.no/projectweb/mrst/modules/"))