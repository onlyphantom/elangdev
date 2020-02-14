test.py
elang (pkg)
    plot (pkg)
        utils (pkg)
            - plot2d(model, words=None, method="PCA")
            - plotsimilar()
            - plotnetwork()
    word2vec (pkg)
        utils (pkg)
            - create_corpus()
            - remove_stopwords()
            - remove_vulgar()
            - remove_region()
            - remove_datetime()
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
import elang.plot
import elang.plot.utils
from elang.plot.utils import plot2d
import elang.word2vec
import elang.word2vec.utils

To demo:
`python elang/plot/utils/embedding.py`

To test plotting functions: 
```py
from elang.plot.utils import plot2d
from gensim.models import Word2Vec
MODEL_PATH = '../path.to.model'
model = Word2Vec.load(MODEL_PATH)
plot2d(model)
```