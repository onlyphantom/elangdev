import sys, os.path
import gensim
from gensim.models import Word2Vec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

def plotCluster(model, words, method="TSNE", n=10, *args, **kwargs):
    embedding_clusters = []
    word_clusters = [] # (7,10)
    for word in words:
        neighbors = [] 
        embeddings = []
        for similar, _ in model.wv.most_similar(word, topn=n):
            neighbors.append(similar)
            embeddings.append(model.wv[similar])
        embedding_clusters.append(embeddings)
        word_clusters.append(neighbors)
    embedding_clusters = np.array(embedding_clusters)  
    cent_n, neigh_n, dim_n = embedding_clusters.shape # 7, 10, 50 (centroid, neighborhood, dimensions)
    print(f"Embedding Cluster", cent_n, neigh_n, dim_n)
    embedding_clusters = embedding_clusters.reshape(cent_n * neigh_n, dim_n) # (70, 50)
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
    print("WordVec Shape:", word_vec.shape) # (7,10,2)
    legendpatches = []
    with plt.style.context("seaborn-pastel"):
        plt.figure(figsize=(7, 5), dpi=180)
        cmx = cm.get_cmap('Pastel1')
        colors = cmx(np.linspace(0,1,len(words))) # (7,4)
        print("Word Clusters", np.array(word_clusters).shape, word_clusters)
        for word, embedding, neighbor, color in zip(words, word_vec, word_clusters, colors):
            x = word_vec[:,:,0]
            y = word_vec[:,:,1]
            print("x", x, '\n')
            print(x.shape) # 7,10
            # plt.scatter(x, y, c=colors.repeat(10, axis=0), alpha=1, label=word)
            plt.scatter(x, y, c=colors.repeat(10, axis=0), alpha=1)

            patchx = mpatches.Patch(color=color, label=word)
            legendpatches.append(patchx)
            # TODO: Add words for neighbors
            for i, word in enumerate(neighbor):
                print(i, word)

        
        plt.legend(handles=legendpatches)
        # print("x1", x[0])
        # print("word_vec[3,:,:]", word_vec[3,:,:])
        # print("word_vec[3,:,0]", word_vec[3,:,0])
        # print(word_clusters)
        # for i, clust in enumerate(word_clusters):
        #     plt.scatter(x, y, c=colors.repeat(10, axis=0), alpha=0.7, label=word_clusters[0])
        #     print("i",i)
        #     print("clust", clust)
        #     for j, c in enumerate(clust):
        #         print("j, c", j, c)
        #         plt.text(word_vec[i,j,0], word_vec[i, j, 1], c, fontsize=4.5, alpha=0.5)
        # plt.legend(loc=4)
        # for label, embeddings, words, color in zip(words, word_vec, word_clusters, colors):
        #     print(color)
        #     plt.scatter(embeddings[:,0], embeddings[:,1], c=color, alpha=0.5, label=label)

    plt.show()


if __name__ == "__main__":
    MODEL_PATH = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        + "/word2vec/model/fin.model"
        #  + "/word2vec/model/demo2d.model"
    )
    model = Word2Vec.load(MODEL_PATH)
    print("Loaded from Path:", MODEL_PATH, "\n", model)
    
    words = ['bca', 'federal', 'dunia', 'dokumen', 'karyawan', 'pejabat', "hukum"]
    
    plotCluster(model, words, method="TSNE", n=10, perplexity=50, early_exaggeration=50, n_iter=2000, random_state=0)