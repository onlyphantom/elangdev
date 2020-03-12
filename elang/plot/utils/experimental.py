import sys, os.path
import gensim
from gensim.models import Word2Vec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plotNeighbours(model, words, k=10, method="TSNE", draggable=False, *args, **kwargs):
    """plotNeighbours Plot and color the `k` nearest neighbors for each word in 2-dimension 

    Create a Matplotlib plot to display word embeddings and their k-nearest neighbors in 2 dimensions, using a specified dimensionality reduction method (`method`) if the word vectors have more than 2 dimensions.
    Set`draggable` to `True` for a draggable legend in the resulting plot. 
    
    Any other parameters specified using `*args` or `**kwargs` is unpacked and passed on to the underlying dimensionality reduction method in `sklearn`.

    :param model: An instance of Word2Vec
    :type model: Word2Vec
    :param words: List of centroid words to render in plot
    :type words: List
    :param k: Number of neighbors for each word, defaults to 10
    :type k: int, optional
    :param method: Method for dimensionality reduction, defaults to "TSNE"
    :type method: str, optional
    :param draggable: Set to `True` if a draggable legend box is preferred, defaults to False
    :type draggable: bool, optional
    :raises AssertionError: Ensure `model` is size 2 (2-dimension word vectors) or higher
    """
    assert (
        model.vector_size >= 2
    ), "This function expects a model of size 2 (2-dimension word vectors) or higher."

    if isinstance(words, str):
        try:
            targets = [targets]
        except TypeError:
            raise TypeError("The targets parameter expect a python list")

    embedding_clusters = []
    word_clusters = [] # (7,10)
    for word in words:
        neighbors = [] 
        embeddings = []
        for similar, _ in model.wv.most_similar(word, topn=k):
            neighbors.append(similar)
            embeddings.append(model.wv[similar])
        embedding_clusters.append(embeddings)
        word_clusters.append(neighbors)
    embedding_clusters = np.array(embedding_clusters)  
    cent_n, neigh_n, dim_n = embedding_clusters.shape # 7, 10, 50 (centroid, neighborhood, dimensions)
    print(f"Embedding Cluster", cent_n, neigh_n, dim_n)
    embedding_clusters = embedding_clusters.reshape(cent_n * neigh_n, dim_n) # (70, 50)
    print(f"Embedding Cluster Reshaped", embedding_clusters.shape)

    if model.vector_size > 2:
        if method == "PCA":
            from sklearn.decomposition import PCA
            word_vec = PCA(2, *args, **kwargs).fit_transform(embedding_clusters)

        elif method == "TSNE":
            from sklearn.manifold import TSNE
            word_vec = TSNE(2, *args, **kwargs).fit_transform(embedding_clusters)

        else:
            raise AssertionError(
                "Model must be one of PCA or TSNE for model with greater than 2 dimensions"
            )

    word_vec = word_vec.reshape(cent_n, neigh_n, -1)
    print("WordVec Shape:", word_vec.shape) # (7,10,2)
    # legendpatches = []
    with plt.style.context("seaborn-pastel"):
        plt.rc('legend', fontsize=7, fancybox=True, framealpha=0.8, facecolor="#777777", edgecolor="#000000")
        plt.rc('font', size=7)
        plt.figure(figsize=(7, 5), dpi=180)
        cmx = cm.get_cmap('Pastel1')
        colors = cmx(np.linspace(0,1,len(words))) # (7,4)
        for word, embedding, neighbor, color in zip(words, word_vec, word_clusters, colors):
            x = embedding[:,0]
            y = embedding[:,1]
            plt.scatter(x, y, color=color, alpha=1, label=word)

            # patchx = mpatches.Patch(color=color, label=word)
            # legendpatches.append(patchx)
            for i, word in enumerate(neighbor):
                plt.annotate(word, alpha=0.6, xy=(x[i], y[i]), size=5)

        if draggable:
            # leg = plt.legend(handles=legendpatches)
            leg = plt.legend()
            leg.set_draggable(state=True)
        else:
            leg = plt.legend(loc="lower left", ncol=min(5, len(words)))
            plt.setp(leg.get_texts(), color='w')
            

    plt.show()


if __name__ == "__main__":
    MODEL_PATH = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        + "/word2vec/model/fin.model"
        #  + "/word2vec/model/demo2d.model"
    )
    # model = Word2Vec.load("/Users/samuel/Downloads/scrape5w500d/scrape5w500d.model") 
    model = Word2Vec.load("/Users/samuel/Datasets/corpus/elang500d.model") 
    print("Loaded from Path:", MODEL_PATH, "\n", model)
    
    words = ['bca', 'hitam', 'hutan', 'pisang', 'mobil', "pulau", "pejabat", "android", "kompas"]
    
    plotNeighbours(model, 
        words, 
        method="TSNE", 
        k=15,
        draggable=True,
        perplexity=12, early_exaggeration=50, n_iter=2000, random_state=0)
