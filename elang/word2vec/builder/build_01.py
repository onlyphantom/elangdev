import os, re, multiprocessing

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

realpath = os.path.dirname(os.path.realpath(__file__))
folderpath = realpath + "/corpus"

##### ##### ##### #####
# BUILD FROM WIKIPEDIA
##### ##### ##### #####
def build_from_wikipedia_random(n=10, lang='id', save=False, model=True, *args, **kwargs):
    """
    Returns a model, and optionally save the results in a subdirectory folder
    """
    articles = []

    url_base = f'https://{lang}.wikipedia.org/wiki/'
    if lang.lower() == 'id':
        random_url = url_base + "Istimewa:Halaman_sembarang"
    elif lang.lower() == 'en':
        random_url = url_base + 'Special:Random'
    else:
        raise ValueError("Please supply one of 'id' or 'en' to the lang argument.")

    for page in tqdm(range(n)):
        url = requests.request("GET", random_url).url
        query = re.sub(url_base, '', url)
        articles.append(_get_wikipedia_article(query, url_base))
    
    
    if save:
        _make_corpus_directory()
        filename = f"wikipedia_random_{n}_{lang}.txt"
        _save_content2txt(articles, filename)
        print("Article contents successfully saved to", filename)

    # build Word2Vec model and save the model
    if model:
        content_list = [d['content'] for d in articles if 'content' in d.keys()]
        print(content_list)
        print(len(content_list))
        corpus = ' '.join(content_list)
        w2vmodel = _create_word2vec(corpus, lang=lang)
        return w2vmodel


def build_from_wikipedia_query(query, save=True):
    pass

def build_from_wikipedia(query=None):
    if query is None:
        build_from_wikipedia_random()
    else:
        build_from_wikipedia_query(query=query)


##### ##### ##### #####
# INTERNAL HELPER FUNCS
##### ##### ##### #####
def _make_corpus_directory():
    path = folderpath + "/txt" 
    if not os.path.exists(path):
        os.makedirs(path)

def _get_wikipedia_article(query, url_base):
    url_query = url_base + str(query)
    req = requests.get(url_query)
    soup = BeautifulSoup(req.content, "html.parser")

    article = {}
    article['title'] = soup.find("h1", attrs={"class": "firstHeading"}).text
    article['url'] = url_query

    find_div = soup.find("div", attrs={"class": "mw-parser-output"})
    if find_div is None:
        return
    for s in find_div(['script', 'style', 'table', 'div']):
        s.decompose()

    find_content = find_div.findAll(["p", "li", "h2.span.mw-headline", "h3.span.mw-headline"])

    article['content'] = ' '.join([re.sub(r'\s+', ' ', row.text) for row in find_content])

    find_redirect_link = find_div.findAll("a", attrs={"class": "mw-redirect"})
    article['related_queries'] = [link['href'][6:] for link in find_redirect_link]
    return article


def _save_content2txt(dictionary, filename):
    content_list = [d['content'] for d in dictionary if 'content' in d.keys()]
    with open(f"{folderpath}/txt/{filename}", "w", encoding = "utf-8") as f:
        f.write("\n".join(content_list))


def _create_word2vec(corpus, lang, size=100, window=5, iteration=10):
    model = Word2Vec(
        corpus,
        size=size,
        window=window,
        min_count=1,
        workers=multiprocessing.cpu_count(),
        iter=iteration,
    )

    model.save(f"{folderpath}/{lang}_{size}d.model")

    return model


##### ##### ##### #####
# RUN DIRECTLY
##### ##### ##### #####

if __name__ == '__main__':
    model = build_from_wikipedia_random(3, lang='id', save=False)
