import os
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import math
import pickle

realpath = os.path.dirname(os.path.realpath(__file__))
folderpath = realpath + "/scrape-results"


def get_tirtoid_urls(query):
    url_base = "https://tirto.id"
    url_query = url_base + "/search?q=" + query
    req = requests.get(url_query)
    soup = BeautifulSoup(req.content, "html.parser")

    # get total page number
    try:
        find_pagination = soup.findAll("li", attrs = {"class": "pagination-item"})
        pagination_list = [row.a.text for row in find_pagination]
        total_page = int(pagination_list[-2])
    except:
        print("Article Not Found")
        return None

    # iterate each page number, to get the title and url
    articles = []
    for page_num in range(1, total_page+1):
        url = url_query + "&p=" + str(page_num)
        r = requests.get(url)
        s = BeautifulSoup(r.content, "html.parser")

        find_article = s.findAll("div", attrs = {"class": "news-list-fade"})
        for row in find_article:
            article = {}
            article['title'] = row.h1.text
            article['url'] = url_base + row.a['href']
            articles.append(article)

        if page_num % 100 == 0:
            print("Page: {} out of {}".format(page_num, total_page))

    # return the dictionary
    return articles


def get_tirtoid_contents(articles, batch_size=None):
    # loop through each stored url
    counter = 0
    for article in articles:
        counter += 1

        # access the article url
        req_article = requests.get(article['url'])
        soup_article = BeautifulSoup(req_article.content, "html.parser")

        # preprocessing html
        for s in soup_article(['script', 'style']):
            s.decompose()
        for br in soup_article.find_all("br"):
            br.replace_with(" ")

        # get article category
        find_category = soup_article.findAll("a", attrs = {"itemprop": "item"})
        article['category'] = find_category[-1].text if len(find_category) else ""

        # get article content (but exclude the "Baca juga" section)
        find_baca_juga_section = soup_article.find("div", attrs = {"class": "baca-holder"})
        try:
            if find_baca_juga_section is not None:
                row.decompose()
        except:
            pass
        
        content = ""
        article_table = soup_article.findAll("div", attrs = {"class": "content-text-editor"})[:-1]
        article['content'] = " ".join([re.sub(r'\s+', ' ', row.text) for row in article_table])

        # save content to file, per batch
        # tsv: category, content, title, url
        # txt: content and title
        if batch_size is not None:
            if counter % batch_size == 0 or counter == len(articles):
                print("Article: {} out of {}".format(counter, len(articles)))

                batch_num = (counter-1) // batch_size
                start_idx = batch_size * batch_num
                end_idx = min(start_idx + batch_size, len(articles))

                articles_batch = articles[start_idx:end_idx]

                filename = "tirtoid_{}_#{}_{}".format(query, batch_num+1, len(articles_batch))

                save_content2tsv(articles_batch, filename + ".tsv")
                save_content2txt(articles_batch, filename + ".txt")

    if batch_size is None:
        filename = "tirtoid_{}_{}".format(query, len(articles))
        save_content2tsv(articles, filename + ".tsv")
        save_content2txt(articles, filename + ".txt")

    return articles


def get_detikcom_urls(query):
    url_base = "https://www.detik.com"
    url_query = url_base + "/search/searchnews?query=" + query
    req = requests.get(url_query)
    soup = BeautifulSoup(req.content, "html.parser")

    # get total page number
    try:
        find_total_article = soup.find("div", attrs = {"class": "search-result"})
        total_article_match = re.search("\\d+", find_total_article.span.text)
        total_article = int(total_article_match.group(0))

        total_page = int(math.ceil(total_article/9))
        total_page = min(1111, total_page) # detik only provides max. 1111 pages
    except:
        print("Article Not Found")
        return None

    # iterate each page number
    articles = []
    for page_num in range(1, total_page+1):
        url = url_query + "&page=" + str(page_num)
        r = requests.get(url)
        s = BeautifulSoup(r.content, "html.parser")

        find_article = s.findAll("article")
        for row in find_article:
            article = {}

            # get url
            article['url'] = row.a['href']

            # get title
            article['title'] = row.h2.text

            # get category
            find_category = row.find("span", attrs = {"class": "category"})
            article['category'] = find_category.text
            find_category.decompose()

            # get posted date
            # article['posted_date'] = row.find("span", attrs = {"class": "date"}).text

            articles.append(article)
            
    # return the dictionary
    return articles


def get_detikcom_contents(articles, batch_size=None):
    # loop through each stored url
    counter = 0
    for article in articles:
        counter += 1

        # access the article url
        try:
            req_article = requests.get(article['url'] + "?single=1")
        except:
            continue
            
        soup_article = BeautifulSoup(req_article.content, "html.parser")

        # preprocessing html
        for s in soup_article(['script', 'style']):
            s.decompose()
        for br in soup_article.find_all("br"):
            br.replace_with(" ")

        # get article content
        content = ""
        find_div = soup_article.find("div", attrs = {"class": "detail__body-text"})
        if find_div is None:
            find_div = soup_article.find("div", attrs = {"class": "itp_bodycontent"})
        if find_div is None:
            find_div = soup_article.find("div", attrs = {"class": "detail_text"})
            
        if find_div is not None:
            article_content = find_div.findAll("p")
            if len(article_content) == 0:
                article_content = [find_div]
            article['content'] = " ".join([re.sub(r'\s+', ' ', row.text) for row in article_content])
        else:
            article['content'] = ""


        # save content to file, per batch
        # tsv: category, content, title, url
        # txt: content and title
        if batch_size is not None:
            if counter % batch_size == 0 or counter == len(articles):
                print("Article: {} out of {}".format(counter, len(articles)))

                batch_num = (counter-1) // batch_size
                start_idx = batch_size * batch_num
                end_idx = min(start_idx + batch_size, len(articles))

                articles_batch = articles[start_idx:end_idx]

                filename = "detikcom_{}_#{}_{}".format(query, batch_num+1, len(articles_batch))

                save_content2tsv(articles_batch, filename + ".tsv")
                save_content2txt(articles_batch, filename + ".txt")

    if batch_size is None:
        filename = "detikcom_{}_{}".format(query, len(articles))
        save_content2tsv(articles, filename + ".tsv")
        save_content2txt(articles, filename + ".txt")

    return articles


def open_pickle(filename):
    with open("{}/pkl/{}".format(folderpath, filename), "rb") as f:
        return pickle.load(f)
    

def save2pickle(filename, l):
    with open("{}/pkl/{}".format(folderpath, filename), 'wb') as f:
        pickle.dump(l, f)


def save_content2tsv(dictionary, filename):
    df = pd.DataFrame(dictionary)
    df.to_csv("{}/tsv/{}".format(folderpath, filename), sep = "\t", index = False)


def save_content2txt(dictionary, filename):
    title_content_list = [d['title'] + "\n" + d['content'] for d in dictionary if 'content' in d.keys()]
    with open("{}/txt/{}".format(folderpath, filename), "w", encoding = "utf-8") as f:
        f.write("\n".join(title_content_list))


def convert_tsv2txt(source_filename, destination_filename):
    df = pd.read_csv("{}/tsv/{}".format(folderpath, source_filename), sep = '\t', encoding = "utf-8")

    title_content_series = df["title"] + "\n" + df["content"]
    with open("{}/txt/{}".format(folderpath, destination_filename), "w", encoding = "utf-8") as f:
        f.write("\n".join([str(row) for row in title_content_series]))


def make_new_folders():
    for filetype in ['tsv', 'txt', 'pkl']:
        path = folderpath + "/" + filetype
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == '__main__':
    make_new_folders()
    query = "tirto"

    # TIRTO.ID
    '''
    # generate new list of dictionary
    tirto = get_tirtoid_urls(query)
    save2pickle("tirtoid_{}.pkl".format(query), tirto)
    '''
    tirto = open_pickle("tirtoid_{}.pkl".format(query))
    tirto_contents = get_tirtoid_contents(tirto)


    # DETIK.COM
    '''
    # generate new list of dictionary
    detik = get_detikcom_urls(query)
    save2pickle("detikcom_{}.pkl".format(query), detik)
    '''
    #detik = open_pickle("detikcom_{}.pkl".format(query))
    #detik_contents = get_detikcom_contents(detik, batch_size = 2500)
