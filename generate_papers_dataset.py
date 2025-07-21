import requests
from bs4 import BeautifulSoup
from pypdf._reader import PdfReader
import io
import time
from typing import List

start_links = ["https://www.sintef.no/projectweb/mrst/publications/papers-by-mrst-team/",
               "https://www.sintef.no/projectweb/mrst/publications/proceedings-by-mrst-team/",
               "https://www.sintef.no/projectweb/mrst/publications/papers-by-others/",
               "https://www.sintef.no/projectweb/mrst/publications/proceedings-papers/",
               "https://www.sintef.no/projectweb/mrst/publications/papers-by-others2/",
               "https://www.sintef.no/projectweb/mrst/publications/phd-theses/",
               "https://www.sintef.no/projectweb/mrst/publications/master-theses/"]

def split_by_more(string: str, chars: List[str] = []):
    if chars == []:
        return string.split()
    else:
        for method, next_method in zip(chars, chars[1:]):
            string = next_method.join(string.split(method))
        return string.split(chars[-1])

def get_authors(og_line):
    title = ""
    last_index = 0
    i = 0
    name_line = ""
    while last_index != -1:
        i+=1
        index = og_line.find('.', last_index+1, -1)
        diff = index-last_index
        if diff >= 30:
            title = og_line[last_index+2:index]
            name_line = og_line[:last_index]
            break
        else:
            last_index = index
        if i > 20:
            break

    names = split_by_more(name_line,[",and","and",","])
    return names, title, name_line


links = []
authors = {}
titles = []
paper_authors = []

for mrst_link in start_links:
    response = requests.get(mrst_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    for section in soup.find_all('div', class_ = "rich-text-field"):
            for article in section.find_all("li"):
                for link_box in article.find_all('a'):
                    paper_link = link_box.get('href')
                    if paper_link != None:
                        if 'http' in paper_link:
                            links.append(paper_link)
                        else:
                            links.append('https://www.sintef.no' + paper_link)

                        line = article.get_text()
                        a, title, name_line = get_authors(line)
                        this_paper_authors = []
                        for single_a in a:
                            single_a = single_a.replace(" ","")
                            split = split_by_more(single_a.strip(), [".","-"])
                            split = [s for s in split if s not in [""," ","  "]]
                            if len(split)>=2:
                                single_a = split[0] + "." + split[-1]
                                authors[single_a] = authors.get(single_a,0) + 1
                                this_paper_authors.append(single_a)

                        paper_authors.append(this_paper_authors)

                        titles.append(title)

def try_fetching_pdf(link):
    response = requests.get(link)
    ctype = response.headers.get("Content-Type")
    if "pdf" in ctype:
        try:
            on_fly_mem_obj = io.BytesIO(response.content)
            return PdfReader(on_fly_mem_obj)
        except:
            pass

    url = response.url

    # Gotta make methods to parse different journals into pdfs

    if "ager" in url:       # Advanced in Geo-Energy research
        pass

    if "springer" in url:
        pass
    
    return None
    
def extract_titles_from_outline(outline):
    result = []
    for item in outline:
        if isinstance(item, list):
            result.extend(extract_titles_from_outline(item))
        else:
            result.append(" ".join(item.title.lower().split()))
    return result
    
x = len(links)
succesfull = 0

for i, (url, a, t) in enumerate(zip(links, paper_authors, titles)):
    print(f"Working with paper nr. {i+1} out of {x}", end = "\r")
    reader = try_fetching_pdf(url)
    if reader != None:
        outline = reader.outline

        sections = extract_titles_from_outline(outline)

        total_text = ""

        for page in reader.pages:
            page_text = " ".join(page.extract_text().lower().split())
            total_text += page_text

        indexes = []
        for i, sec in enumerate(sections):
            if i == 0:
                start_index = 0
            else:
                start_index = indexes[i-1]
            
            indexes.append(total_text.find(sec, start_index))

        chapters = []

        for i_start, i_end, sec in zip(indexes, indexes[1:], sections):
            s = total_text[i_start:i_end]
            s = s.replace(sec, sec + ":\n\n", count = 1)
            try:
                n = len(s)
                for i in range(2,10):

                    if s[n-i] == " ":
                        num_sec = s[n-i:]
                        s = s.replace(num_sec, "")
                        break
            except:
                pass

            chapters.append(s)

        succesfull += 1

    if i%10==0:
        print(f"{succesfull} successfull out of {i+1}", end = "\r")
        time.sleep(5)