from gensim.models import Word2Vec
import cPickle as pickle
import pandas as pd
import numpy as np
from scipy.sparse import find as sparse_find
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from itertools import combinations
import seaborn

# http://www.foodsubs.com/WinesRed.html


def read_files(names):

    for name in names:
        tmp = pd.read_csv(
            '../../data/dataframes/' + name)

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    return df


def load_files():
    ###
    # import word2vec model
    model = Word2Vec.load("../eda/wine_word2vec_model.model")

    # import pickled files
    f = open('pickle_files/cv.pkl', 'r')
    cv = pickle.load(f)
    f. close()

    f = open('pickle_files/tf.pkl', 'r')
    tf = pickle.load(f)
    f. close()

    f = open('pickle_files/tfidf.pkl', 'r')
    tfidf = pickle.load(f)
    f. close()

    # read in data from cleaned csvs
    reviews = read_files(
        ['first_1000.csv', 'next_9000.csv', 'remainder_reviews.csv'])
    reviews = reviews[~reviews['review_text'].isnull()]

    # get name labels into numpy array
    y = reviews['wine_name']

    # unique name entries
    names = y.unique()

    return model, cv, tf, tfidf, names


def map_wine_vectors(tfidf, model, names, col_dict):
    vec_dim = model.syn0.shape[1]

    wine_vectors = np.zeros((len(names), vec_dim))

    for i in range(len(names)):

        row = tfidf[i, :]
        (_, j_s, vals) = sparse_find(row)

        vecs = np.zeros((len(j_s), vec_dim))

        for k, j in enumerate(j_s):
            if col_dict[j] in model:
                vecs[k, :] = model[col_dict[j]]
        print i
        vec = np.average(vecs, axis=0, weights=vals)

        wine_vectors[i, :] = vec

    return wine_vectors


def plot_dim_red(x_red, clusts=None, x=0, y=1, z=2):
    colors = ['k', 'r', 'g', 'c', 'm', 'y', 'b', 'brown', 'gold', 'brown']

    if clusts is not None:
        for i in np.unique(clusts):
            x_clust = x_red[clusts == i, :]

            plt.scatter(x_clust[:, x], x_clust[:, y],
                        color=colors[i],
                        label=i,
                        alpha=0.5)
        plt.legend(loc='best')
        plt.show()
    else:
        plt.scatter(x_red[:, x], x_red[:, y], c=x_red[:, z],
                    alpha=0.6,
                    cmap=plt.get_cmap('hot'))
        plt.show()


def plot_dist_hists(wine_vectors, names, varietal1, varietal2):
    # find indices in names matching varietal1 and varietal2
    var1 = np.where(map(lambda x: (varietal1 in x) * 1, names))[0]
    var2 = np.where(map(lambda x: (varietal2 in x) * 1, names))[0]

    combs_var1 = list(combinations(var1, 2))

    combs_inter = []

    for i in var1:
        for j in var2:
            combs_inter.append((i, j))

    dists_var1 = []

    dists_inter = []

    for (i, j) in combs_var1:
        dists_var1.append(cosine(wine_vectors[i, :], wine_vectors[j, :]))

    for (i, j) in combs_inter:
        dists_inter.append(cosine(wine_vectors[i, :], wine_vectors[j, :]))

    plt.hist(dists_var1, normed=True, bins=50, alpha=0.5,
             label="{} - {} Cosine Dists".format(varietal1, varietal1))
    plt.hist(dists_inter, normed=True, bins=50, alpha=0.5,
             label="{} - {} Cosine Dists".format(varietal1, varietal2))
    plt.legend(loc="best")
    plt.title('Inter- and Intra- Varietal Cosine Distance')
    plt.show()

if __name__ == "__main__":
    model, cv, tf, tfidf, names = load_files()

    # remove entries with no tfidf weights
    non_zero = np.where(tfidf.sum(axis=1) != 0)[0]
    tfidf = tfidf[non_zero, :]
    names = names[non_zero]

    words = cv.vocabulary_.keys()
    cols = cv.vocabulary_.values()

    col_dict = dict(zip(cols, words))

    # take list of names and tfidf vector
    # average up word vectors using tfidf weights to project each wine
    # into word2vec space
    wine_vectors = map_wine_vectors(tfidf, model, names, col_dict)

    # cluster and visualize using PCA/TSNE
    # to verify that the word2vec doesn't blow up the tfidf result

    # cluster using k-means
    km = KMeans()

    clusts = km.fit_predict(wine_vectors)

    # latent semantic analysis (SVD on TFIDF)
    svd = TruncatedSVD(n_components=50)

    x_svd = svd.fit_transform(wine_vectors)

    # plot the X and Y principal components
    plot_dim_red(x_svd, clusts, x=0, y=1)

    # calculate tsne
    tsne = TSNE(n_components=2)

    x_tsne = tsne.fit_transform(x_svd)

    # visualize top clusters with tsne
    plot_dim_red(x_tsne, clusts, x=0, y=1)

    cabs = map(lambda x: ('Cabernet' in x) * 1, names)
    merlots = map(lambda x: ('Merlot' in x) * 1, names)
    pinots = map(lambda x: ('Pinot Noir' in x) * 1, names)
    champagnes = map(lambda x: ('Champagne' in x) * 1, names)
    rieslings = map(lambda x: ('Riesling' in x) * 1, names)
    syrahs = map(lambda x: ('Syrah' in x) * 1, names)
    zins = map(lambda x: ('Zinfandel' in x) * 1, names)
    sauv = map(lambda x: ('Sauvignon Blanc' in x) * 1, names)
    chards = map(lambda x: ('Chardonnay' in x) * 1, names)
    blancs = map(lambda x: ('Blanc' in x) * 1, names)
    chablis = map(lambda x: ('Chablis' in x) * 1, names)

    plot_dim_red(x_svd, chablis, x=0, y=1)
    # predict!
    # 3, 4, 6, 59
    # cosine(wine_vectors[3, :], wine_vectors[4,:])
    # 82, 97, 100, 7182, 7187
    wv = wine_vectors[3, :]  # search vector
    dists = np.apply_along_axis(lambda x: cosine(x, wv),
                                axis=1,
                                arr=wine_vectors)

    # using list of words
    # add them up, take cosine similarity across all wines
    terms = ['dark', 'fruit', 'oak', 'tannin']

    # using wine in database, with option to add or subtract qualities
    # create search tool for database
    # add up vectors, taking cosine similarity across all wines

    # https://github.com/seatgeek/fuzzywuzzy to do fuzzy term matching?
