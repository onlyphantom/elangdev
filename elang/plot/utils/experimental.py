import sys, os.path
import gensim
from gensim.models import Word2Vec

import numpy as np
import matplotlib.pyplot as plt

def plotCluster(model, words, method="PCA", n=10, *args, **kwargs):
    embedding_clusters = []
    word_clusters = []
    for word in words:
        neighbors = []
        embeddings = []
        for similar, _ in model.most_similar(word, topn=n):
            neighbors.append(similar)
            embeddings.append(model.wv[word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)   
    cent_n, neigh_n, dim_n = embedding_clusters.shape # 7, 10, 50 (centroid, neighborhood, dimensions)
    print(f"Embedding Cluster", cent_n, neigh_n, dim_n)
    embedding_clusters = embedding_clusters.reshape(cent_n * neigh_n, dim_n)
    print(f"Embedding Cluster Reshaped", embedding_clusters.shape)

    if method == "PCA":
        from sklearn.decomposition import PCA
        word_vec = PCA(2, *args, **kwargs).fit_transform(embedding_clusters)

    elif method == "TSNE":
        from sklearn.manifold import TSNE
        # TODO: optional kwargs: perplexity (k-neighbors), early_exaggeration etc
        # Refer to: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        word_vec = TSNE(2, *args, **kwargs).fit_transform(embedding_clusters)

    else:
        raise AssertionError(
            "Model must be one of PCA or TSNE for model with greater than 2 dimensions"
        )

    word_vec = word_vec.reshape(cent_n, neigh_n, -1)
    print("WordVec Shape:", word_vec.shape)


if __name__ == "__main__":
    MODEL_PATH = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        + "/word2vec/model/fin.model"
        #  + "/word2vec/model/demo2d.model"
    )
    model = Word2Vec.load(MODEL_PATH)
    print("Loaded from Path:", MODEL_PATH, "\n", model)
    
    words = ['bca', 'indonesia', 'dunia', 'dokumen', 'karyawan', 'pejabat', 'era']
    
    plotCluster(model, words)