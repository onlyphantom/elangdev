import os
from gensim.models import Word2Vec
from simple_preprocess.bcapara import single_para, multi_senc_demo

SIZE = 3
WINDOW = 2
ITER = 10

# build vocabulary and train model on one paragraph
# corpus = single_para()
corpus = multi_senc_demo()
model = Word2Vec(
    corpus,
    seed=100,
    size=SIZE,
    window=WINDOW,
    min_count=2,
    iter=ITER
)

print(model)
print(model.wv.most_similar('bank'))