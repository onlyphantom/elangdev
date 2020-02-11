import os, multiprocessing
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from utils import remove_stopwords_id
import pickle

SIZE = 100
WINDOW = 5
ITER = 1000

REAL_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_DIR = REAL_PATH + "/scraper/scrape-results/txt/"
MODEL_DIR = REAL_PATH + "/model/scrape{}w{}d.model".format(WINDOW, SIZE)

def combine_txt():
    list_txt = [f for f in os.listdir(FILE_DIR) if f.endswith(".txt")]
    content = [open(FILE_DIR + f, encoding="utf-8").read().splitlines() for f in list_txt]
    flat_content = [c for sublist in content for c in sublist]
    return flat_content


def create_corpus(sentences, save=True):
    sentences = list(map(remove_stopwords_id, sentences))
    print(len(sentences))
    corpus = list(map(simple_preprocess, sentences))
    print(corpus[:2], "\nSentences: -->", len(corpus))
    uniqset = set(word for l in corpus for word in l)
    print(len(uniqset), "Unique Terms")

    if save:
        with open("corpus/corpus_{}.pkl".format(len(uniqset)), "wb") as f:
            pickle.dump(corpus, f)
            
    return corpus


def create_word2vec(filename, save=True):
    with open("corpus/{}.pkl".format(filename), "rb") as f:
        corpus = pickle.load(f)
    flat_corpus = [c for sublist in corpus for c in sublist]
    token_size = len(flat_corpus)
    min_count = int(0.001/100 * token_size) + 1
    
    model = Word2Vec(
        corpus,
        size=SIZE,
        window=WINDOW,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        iter=ITER,
    )
    if save:
        model.save(MODEL_DIR)
        print("Model Saved:", MODEL_DIR)

    return model

def load_model(filename):
    return Word2Vec.load(REAL_PATH + "/model/" + filename)


if __name__ == "__main__":
    # sentences = combine_txt()
    # corpus = create_corpus(sentences)
    # model = create_word2vec("corpus_108957")

    model = load_model("scrape5w100d.model")


# print(model.wv.most_similar("bank")
'''
[('perbankan', 0.6497560143470764),
 ('kredit', 0.5568206906318665),
 ('tabungan', 0.5111026167869568),
 ('nasabah', 0.5071349143981934),
 ('bi', 0.50611412525177),
 ('bni', 0.49788159132003784),
 ('bri', 0.4970148503780365),
 ('atm', 0.4831470847129822),
 ('imf', 0.4580811858177185),
 ('rekening', 0.4465577006340027)]
'''
# len(model.wv.vocab) -> 10835
# model.corpus_total_words -> 4656126
# model.wv.vocab['bank'].count -> 1536
