import os, re, multiprocessing

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

realpath = os.path.dirname(os.path.realpath(__file__))
folderpath = realpath + "/corpus"

##### ##### ##### #####
# BUILD FROM WIKIPEDIA
##### ##### ##### #####
def build_from_wikipedia_random(
    n=10, lang="id", save=False, model=True, *args, **kwargs
):
    """build_from_wikipedia_random Builds a corpus from random articles on Wikipedia, optionally training, and returning, a Word2Vec model if `model` is set to True.
    
    Builds a corpus from `n` articles from Wikipedia (either English or Indonesian version, defined by `lang`). When `save` is True, the corpus is saved in the `/corpus` directory. When `model` is True, a Word2Vec model is trained on the corpus and returned.
    
    Any other parameters specified using `*args` or `**kwargs` is unpacked and passed on to the underlying `Word2Vec` method from gensim.

    :param n: The number of random articles to parse, defaults to 10
    :type n: int, optional
    :param lang: The language version of Wikipedia to parse from (`en` for English, `id` for Indonesian), defaults to "id"
    :type lang: str, optional
    :param save: Save the built corpus in the `/corpus` directory, defaults to False
    :type save: bool, optional
    :param model: Train and return a Word2Vec model on the built corpus, defaults to True
    :type model: bool, optional
    :raises ValueError: `Lang` be one of 'id' or 'en'
    :return: A Word2Vec model trained on the built corpus when `model` is True
    :rtype: Word2Vec model
    """
    articles = []

    url_base = f"https://{lang}.wikipedia.org/wiki/"
    if lang.lower() == "id":
        random_url = url_base + "Istimewa:Halaman_sembarang"
    elif lang.lower() == "en":
        random_url = url_base + "Special:Random"
    else:
        raise ValueError("Please supply one of 'id' or 'en' to the lang argument.")

    for page in tqdm(range(n)):
        url = requests.request("GET", random_url).url
        query = re.sub(url_base, "", url)
        articles.append(_get_wikipedia_article(query, url_base))

    if save:
        _make_corpus_directory()
        filename = f"wikipedia_random_{n}_{lang}.txt"
        _save_content2txt(articles, filename)
        print("Article contents successfully saved to", filename)

    if model:
        content_list = [d["content"] for d in articles if "content" in d.keys()]
        print(content_list)
        print(len(content_list), "\n", "-----")
        # corpus = ' '.join(content_list)
        corpus = list(map(simple_preprocess, content_list))
        print(corpus)
        w2vmodel = _create_word2vec(corpus, lang=lang, *args, **kwargs)
        return w2vmodel


def build_from_wikipedia_query(
    query, levels=5, lang="id", save=True, model=True, *args, **kwargs
):
    articles = []
    url_base = f"https://{lang}.wikipedia.org/wiki/"
    try:
        article = _get_wikipedia_article(query, url_base)
        articles.append(article)
        related_queries = list(set(article["related_queries"]))
        all_queries = [query]
        print("Related Queries", related_queries)
        print("All Queries", all_queries)
    except:
        raise Exception("no article found, try another query")


def build_from_wikipedia(query=None, n=10, lang="id", *args, **kwargs):
    if query is None:
        build_from_wikipedia_random(lang=lang, n=n, *args, **kwargs)
    else:
        build_from_wikipedia_query(query=query, lang=lang, *args, **kwargs)


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
    article["title"] = soup.find("h1", attrs={"class": "firstHeading"}).text
    article["url"] = url_query

    find_div = soup.find("div", attrs={"class": "mw-parser-output"})
    if find_div is None:
        return
    for s in find_div(["script", "style", "table", "div"]):
        s.decompose()

    find_content = find_div.findAll(
        ["p", "li", "h2.span.mw-headline", "h3.span.mw-headline"]
    )

    article["content"] = " ".join(
        [re.sub(r"\s+", " ", row.text) for row in find_content]
    )

    find_redirect_link = find_div.findAll("a", attrs={"class": "mw-redirect"})
    article["related_queries"] = [link["href"][6:] for link in find_redirect_link]
    return article


def _save_content2txt(dictionary, filename):
    content_list = [d["content"] for d in dictionary if "content" in d.keys()]
    with open(f"{folderpath}/txt/{filename}", "w", encoding="utf-8") as f:
        f.write("\n".join(content_list))


def _create_word2vec(corpus, lang, size=100, window=5, iteration=10, min_count=1):
    model = Word2Vec(
        corpus,
        size=size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        iter=iteration,
    )

    model.save(f"{folderpath}/{lang}_{size}d.model")

    return model


##### ##### ##### #####
# RUN DIRECTLY
##### ##### ##### #####

if __name__ == "__main__":
    # model = build_from_wikipedia_random(n=3, lang="id", save=True, size=20, min_count=3)
    # model = build_from_wikipedia(n=3, lang="id", save=True, size=20, min_count=3)
    model = build_from_wikipedia(
        query="Koronavirus", lang="id", save=True, size=20, min_count=3
    )

