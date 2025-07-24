import requests
from bs4 import BeautifulSoup
from pypdf._reader import PdfReader
import io
from typing import List

from dotenv import load_dotenv
import os
load_dotenv()
elsevier_api_key = os.getenv("ELSEVIER_API_KEY")

visited = set()

###### EVERYTHING THAT IS RETURNED AS NONE IN THE TRY FETCHING DEFINITION HAS TO BE IMPLEMENTED IN A NICE WAY TO GATHER THE DATA AND CORRECT DATA FORMAT

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

def try_fetching_pdf(url, only_abstract):

    """
    Tries fetching pdf from url. If we see no way to get a pdf of the article from the url,
    we return the article in dict form.
    If return PDFREADER -> only_abstract gets handled outside of this function, returns the same no matter only_abstract
    If not PDFREADER:
        If only_abstract:
            returns dict{
            'abstract': <THE ABSTRACT>,
            ('keywords': <[LIST OF KEYWORDS]>) -- POSSIBLY (NOT ALWAYS)
            }
    If could not read text, return None
    """

    try:
        response = requests.get(url)
        ctype = response.headers.get("Content-Type")
        try:
            if "pdf" in ctype:
                on_fly_mem_obj = io.BytesIO(response.content)
                return PdfReader(on_fly_mem_obj)
        except:
            pass
        
        r_url = response.url
        https = "https://"
        url_snippet = r_url[len(https):]
        index = url_snippet.find('/')
        url_snippet = url_snippet[:index]

    except Exception as e:
        return None

    actually_try = True
    is_common_one = True

    if "linkinghub.elsevier.com" == url_snippet:
        """
        Works as for now only with only_abstract, but should probably get the other one to work as well
        """

        if actually_try:

            if only_abstract:

                https_doi = "https://doi.org"
                doi = url[len(https_doi):]
                elsevier_url = "https://api.elsevier.com/content/article/doi/" + doi + "?view=META_ABS"
                response = requests.get(elsevier_url, headers={"X-ELS-APIKey": elsevier_api_key})
                soup = BeautifulSoup(response.content, 'html.parser')
                abstract = soup.get_text().split("\n")[1].strip()
                return {"abstract": abstract}

            else:

                # THIS ONE DOES NOT WORK ATM

                https_doi = "https://doi.org"
                doi = url[len(https_doi):]
                elsevier_url = "https://api.elsevier.com/content/article/doi/" + doi + "?view=FULL"
                response = requests.get(elsevier_url, headers={"X-ELS-APIKey": elsevier_api_key})
                return None

    elif "onepetro.org" == url_snippet:
        """
        Returnd dict if only_abstract, None else
        """

        if actually_try:

            if only_abstract:

                # SHOULD WAIT FOR LEGAL THINGS

                from selenium.webdriver import Firefox
                from selenium.webdriver.firefox.options import Options
                from selenium.webdriver.firefox.service import Service

                firefox_path = r"/home/aasmund_mjos/firefox_mozilla/firefox"
                geckodriver_path = r"/home/aasmund_mjos/Downloads/geckodriver-v0.36.0-linux64/geckodriver"

                service = Service(geckodriver_path)
                options = Options()
                options.binary_location = firefox_path

                driver = Firefox(service = service, options=options)
                driver.get(r_url)
                html = driver.page_source

                soup = BeautifulSoup(html, 'html.parser')
                abstract_box = soup.find("section", class_ = 'abstract')
                abstract = ""
                for p in abstract_box.find_all('p'):
                    abstract += p.get_text()
                    abstract += "\n"

                driver.close()
                return {"abstract": abstract}

            else:
                pass

    elif "link.springer.com" == url_snippet:
        """
        Returns dict if only_abstract, None else
        """
        if actually_try:
            if only_abstract:

                soup = BeautifulSoup(response.content, 'html.parser')
                abstract_box = soup.find('div', class_ = "c-article-section")
                return {"abstract": abstract_box.get_text()}
        else:

            pass

    elif "www.earthdoc.org" == url_snippet:
        """
        Returns dict if only_abstract, None Else
        """
        if actually_try:
            if only_abstract:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Accept": "text/html",
                }
                response = requests.get(r_url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                abstract = ""
                abstract_box = soup.find("div", class_ = "description")
                for p in abstract_box.find_all('p'):
                    abstract += p.get_text()
                    abstract += "\n"
                return {"abstract": abstract}
            else:
                pass

    elif "www.mdpi.com" == url_snippet:
        """
        Always returns PDFREADER
        """
        if actually_try:

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "text/html",
            }
            response = requests.get(r_url + '/pdf', headers=headers)
            ctype = response.headers.get("Content-Type")
            if "pdf" in ctype:
                on_fly_mem_obj = io.BytesIO(response.content)
                reader = PdfReader(on_fly_mem_obj)
                return reader
            else:
                return None

    elif "agupubs.onlinelibrary.wiley.com" == url_snippet:

        """
        Always returns dict
        """

        if actually_try:
            if only_abstract:
                from selenium.webdriver import Firefox
                from selenium.webdriver.firefox.options import Options
                from selenium.webdriver.firefox.service import Service

                firefox_path = r"/home/aasmund_mjos/firefox_mozilla/firefox"
                geckodriver_path = r"/home/aasmund_mjos/Downloads/geckodriver-v0.36.0-linux64/geckodriver"

                service = Service(geckodriver_path)
                options = Options()
                options.binary_location = firefox_path

                driver = Firefox(service = service, options=options)
                driver.get(r_url)
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                abstract = ""

                for p in soup.find_all('p'):
                    span = p.find('span', class_='paraNumber')
                    if span:
                        abstract = p.get_text()
                        break
                        
                driver.quit()
                return {"abstract": abstract}
            
            else:
                from selenium.webdriver import Firefox
                from selenium.webdriver.firefox.options import Options
                from selenium.webdriver.firefox.service import Service

                firefox_path = r"/home/aasmund_mjos/firefox_mozilla/firefox"
                geckodriver_path = r"/home/aasmund_mjos/Downloads/geckodriver-v0.36.0-linux64/geckodriver"

                service = Service(geckodriver_path)
                options = Options()
                options.binary_location = firefox_path

                driver = Firefox(service = service, options=options)
                driver.get(r_url)
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                paragraphs = []

                for p in soup.find_all('p'):
                    span = p.find('span', class_='paraNumber')
                    if span:
                        paragraphs.append(p.get_text())

                driver.quit()
                return {"sections": paragraphs}

    elif "arxiv.org" == url_snippet:

        """
        Always returns PDFREADER
        """

        if actually_try:

            pdf_link = ""
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup.find_all('a', class_ = "abs-button download-pdf", href = True):
                if "View PDF" in element.get_text():
                    pdf_link = element.get('href')
                    break

            if pdf_link == "":
            
                for element in soup.find_all('a', class_ = "mobile-submission-download", href = True):
                    if "View PDF" in element.get_text():
                        pdf_link = element.get('href')
                        break

            if pdf_link == "":
                return None
            
            else:
                try:
                    response = requests.get(pdf_link)
                    ctype = response.headers.get("Content-Type")
                    if "pdf" in ctype:
                        on_fly_mem_obj = io.BytesIO(response.content)
                        reader = PdfReader(on_fly_mem_obj)
                        return reader
                except:
                    pass
                return None

    elif "ager.yandypress.com" == url_snippet:

        """
        Returns dict{"abstract": <THE ABSTRACT>} if only_abstract
        and PDDREADER if whole document
        """

        if actually_try:
            if only_abstract:

                soup = BeautifulSoup(response.content, 'html.parser')
                abstract = ""
                abstract_box = soup.find('section', class_ = "item abstract")
                for p in abstract_box.find_all('p'):
                    if not "document type:" in p.get_text().lower():
                        abstract += p.get_text()
                        abstract += "\n"
                    else:
                        break

                return {"abstract": abstract}
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
                for element in soup.find_all(href = True):
                    if "PDF" in element.get_text():
                        link = element.get('href')
                        response = requests.get(link)
                        ctype = response.headers.get("Content-Type")
                        soup = BeautifulSoup(response.content, 'html.parser')
                        for element in soup.find_all('a', class_ = "download"):
                            download_link = element.get('href')
                            response = requests.get(download_link)
                            ctype = response.headers.get("Content-Type")
                            if "pdf" in ctype:
                                try:
                                    on_fly_mem_obj = io.BytesIO(response.content)
                                    return PdfReader(on_fly_mem_obj)
                                except:
                                    return None

    else:
        is_common_one = False
    
    if is_common_one:
        if url_snippet not in visited:
            print(url_snippet, ":   ", url)
            visited.add(url_snippet)

    return None
    
def extract_titles_from_outline(outline):
    result = []
    for item in outline:
        if isinstance(item, list):
            result.extend(extract_titles_from_outline(item))
        else:
            result.append(" ".join(item.title.lower().split()))
    return result

def read_pdfs(reader):
    """
    Returns the chapters in the pdf's outline as a list. If no outline found in the pdf document,
    return list with one element of the whole text
    """

    # THIS ONE DOESN'T RETURN ANYTHING AS OF NOW
        
    outline = reader.outline
    sections = extract_titles_from_outline(outline)
    total_text = ""

    for page in reader.pages:
        page_text = " ".join(page.extract_text().lower().split())
        total_text += page_text

    if not bool(outline):
        return [total_text]

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

    return chapters

def generate_dataset(only_abstract = False):

    start_links = ["https://www.sintef.no/projectweb/mrst/publications/papers-by-mrst-team/",
               "https://www.sintef.no/projectweb/mrst/publications/proceedings-by-mrst-team/",
               "https://www.sintef.no/projectweb/mrst/publications/papers-by-others/",
               "https://www.sintef.no/projectweb/mrst/publications/proceedings-papers/",
               "https://www.sintef.no/projectweb/mrst/publications/papers-by-others2/",
               "https://www.sintef.no/projectweb/mrst/publications/phd-theses/",
               "https://www.sintef.no/projectweb/mrst/publications/master-theses/"]

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
        
    x = len(links)
    succesfull = 0

    for i, (url, a, t) in enumerate(zip(links, paper_authors, titles)):
        print(f"Working with paper nr. {i+1} out of {x}", end = "\r")

        reader = try_fetching_pdf(url, only_abstract)
        if reader != None:
            print("Woho")
            if type(reader) == type(PdfReader):
                read_pdfs(reader)

            if type(reader) == type(dict()):
                pass #GOTTA IMPLEMENT

        # if i%10==0:
        #     print(f"{succesfull} successfull out of {i+1}", end = "\r")
        #     time.sleep(5)

reader = try_fetching_pdf('https://doi.org/10.1007/978-3-030-95157-3_21', only_abstract=True)
# generate_dataset(only_abstract=True)