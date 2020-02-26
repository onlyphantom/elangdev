test.py
elang (pkg)
    plot (pkg)
        utils (pkg)
            - plot2d(model, words=None, method="TSNE", targets=[], **kwargs)
            - plotNeighbours(model, words, k=10, method="TSNE", draggable=False, *args, **kwargs)
            - plotNetwork(model, words, k=10, draggable=False)
    word2vec (pkg)
        utils (pkg)
            - extract_detikcom()
            - extract_wikipedia()
            - create_corpus()
            - remove_stopwords(sentence)
            - remove_vulgar(sentence)
            - remove_region(sentence)
            - remove_datetime(sentence)
            - negative (data)
                - stopwords-id.gzip
                - indonesian-region.gzip
                - swear-words.gzip
        model (data)
            indo10k.model  
        corpus (data)
            wiki_banks.gzip
            news_banks.gzip
            news_general.gzip

import elang
from elang.plot.utils import plot2d
import elang.word2vec.utils

To demo:
`python elang/plot/utils/embedding.py`

To test plotting functions: 
```py
from elang.plot.utils import plot2d
from gensim.models import Word2Vec
MODEL_PATH = '../path.to.model'
# MODEL_PATH = '/Users/samuel/Datasets/corpus/demo50d.model'
model = Word2Vec.load(MODEL_PATH)
plot2d(model)
plot2d(model, method="TSNE")
plot2d(
    model,
    method="PCA",
    words=[
        "bca",
        "mandiri",
        "uob",
        "algoritma",
        "nonsense"
    ],
     targets=['uob', 'mandiri','bca']
)

bca = model.wv.most_similar("bca", topn=14)
similar_bca = [w[0] for w in bca] + ["bca"]
plot2d(
    model,
    method="TSNE",
    targets=similar_bca,
    perplexity=20,
    early_exaggeration=50,
    n_iter=2000,
    random_state=0,
)
```