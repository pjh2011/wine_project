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

    f = open('pickle_files/wine_vectors.pkl', 'r')
    wine_vectors = pickle.load(f)
    f. close()

    # read in data from cleaned csvs
    reviews = read_files(
        ['first_1000.csv', 'next_9000.csv', 'remainder_reviews.csv'])
    reviews = reviews[~reviews['review_text'].isnull()]

    # get name labels into numpy array
    y = reviews['wine_name']

    # unique name entries
    names = y.unique()

    return model, cv, tf, tfidf, names, wine_vectors


def print_top_matrix_words(cv_obj, term_matrix, n_words):
    keys = cv_obj.vocabulary_.keys()
    vals = cv_obj.vocabulary_.values()

    word_lookup = dict(zip(vals, keys))

    if isinstance(term_matrix, np.ndarray):
        term_matrix = np.asmatrix(term_matrix)

    for i in range(term_matrix.shape[0]):
        ranked_words = np.fliplr(np.argsort(term_matrix[i, :]))

        print "#################"
        print "TOPIC NUMBER: ", i
        print "#################"

        for j in range(n_words):
            print word_lookup[ranked_words[0, j]]

        print '\n'


def plot_dim_red(x_red, clusts=None, labels=None, title=None, x=0, y=1, z=2,
                 colors=['0.3', 'r', 'g', 'b', 'm', 'y', 'c', 'brown', 'gold',
                         'brown'], alpha=0.3):

    if clusts is not None:
        for i in sorted(np.unique(clusts)):
            x_clust = x_red[clusts == i, :]

            if labels is not None:
                l = labels[i]
            else:
                l = i

            plt.scatter(x_clust[:, x], x_clust[:, y],
                        color=colors[i],
                        label=l,
                        alpha=alpha)
        plt.legend(loc='best')

        if title is not None:
            plt.title(title)

        plt.show()
    else:
        plt.scatter(x_red[:, x], x_red[:, y], c=x_red[:, z],
                    alpha=0.2,
                    cmap=plt.get_cmap('hot'))

        if title is not None:
            plt.title(title)

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
    plt.xlabel("Cosine Distance")
    plt.ylabel("Probability Density")
    plt.show()

if __name__ == "__main__":

    model, cv, tf, tfidf, names, wine_vectors = load_files()

    # cluster and visualize using PCA/TSNE
    # to verify that the word2vec doesn't blow up the tfidf result

    # latent semantic analysis (SVD on TFIDF)
    svd = TruncatedSVD(n_components=50)

    x_svd = svd.fit_transform(wine_vectors)

    # calculate tsne
    random_subset = np.sort(np.random.choice(range(30797), 6000))
    x_svd_sample = x_svd[random_subset, :]

    tsne = TSNE(n_components=2)

    x_tsne_sample = tsne.fit_transform(x_svd_sample)

    # cabs = map(lambda x: ('Cabernet' in x) * 1, names)
    # merlots = map(lambda x: ('Merlot' in x) * 1, names)
    # pinots = map(lambda x: ('Pinot Noir' in x) * 1, names)
    # champagnes = map(lambda x: ('Champagne' in x) * 1, names)
    # rieslings = map(lambda x: ('Riesling' in x) * 1, names)
    # syrahs = map(lambda x: ('Syrah' in x) * 1, names)
    # zins = map(lambda x: ('Zinfandel' in x) * 1, names)
    # sauv = map(lambda x: ('Sauvignon Blanc' in x) * 1, names)
    # chards = map(lambda x: ('Chardonnay' in x) * 1, names)
    # blancs = map(lambda x: ('Blanc' in x) * 1, names)
    # chablis = map(lambda x: ('Chablis' in x) * 1, names)
    #
    # plot_dim_red(x_tsne_sample, np.array(cabs)[random_subset],
    #              x=0, y=1, colors=['0.3', 'r'], labels=[
    #              'Other', 'Cabernet Sauvignon'],
    #              title="Wine Vectors T-SNE Visualization",
    #              alpha=0.5)
    #
    # plot_dim_red(x_tsne_sample, np.array(merlots)[random_subset],
    #              x=0, y=1, colors=['0.3', 'r'], labels=[
    #              'Other', 'Merlot'],
    #              title="Wine Vectors T-SNE Visualization",
    #              alpha=0.5)
    #
    # plot_dim_red(x_tsne_sample, np.array(pinots)[random_subset],
    #              x=0, y=1, colors=['0.3', 'r'], labels=[
    #              'Other', 'Pinot Noir'],
    #              title="Wine Vectors T-SNE Visualization",
    #              alpha=0.5)
    #
    # plot_dim_red(x_tsne_sample, np.array(chards)[random_subset],
    #              x=0, y=1, colors=['0.3', 'gold'], labels=[
    #              'Other', 'Chardonnay'],
    #              title="Wine Vectors T-SNE Visualization",
    #              alpha=0.5)
    #
    # plot_dim_red(x_tsne_sample, np.array(rieslings)[random_subset],
    #              x=0, y=1, colors=['0.3', 'gold'], labels=[
    #              'Other', 'Riesling'],
    #              title="Wine Vectors T-SNE Visualization",
    #              alpha=0.5)
    #
    # plot_dim_red(x_tsne_sample, np.array(sauv)[random_subset],
    #              x=0, y=1, colors=['0.3', 'gold'], labels=[
    #              'Other', 'Sauvignon Blanc'],
    #              title="Wine Vectors T-SNE Visualization",
    #              alpha=0.5)
    #
    # plt.scatter(x_tsne_sample[:, 0], x_tsne_sample[
    #             :, 1], color='0.3', alpha=0.5)
    # plt.title('Wine Vectors T-SNE Visualization')
    # plt.show()

    # NMF
    nmf = NMF(n_components=4)
    doc_matrix_nmf = nmf.fit_transform(tfidf)

    term_matrix_nmf = nmf.components_

    print_top_matrix_words(cv, term_matrix_nmf, 10)

    # pull out top topic from each point in nmf doc matrix
    nmf_top_topics = np.argsort(doc_matrix_nmf, axis=1)[:, -1]

    plot_dist_hists(wine_vectors, names, "Cabernet", "Champagne")

    # plot_dim_red(x_tsne_sample, nmf_top_topics[random_subset],
    #              x=0, y=1, labels=[
    #              'Dark Reds', 'White Wines', 'Light Reds',
    #              'Tannic Reds'],
    #              colors=['r', 'y', 'g', 'b'],
    #              title="Wine Vectors T-SNE Visualization",
    #              alpha=0.3)
