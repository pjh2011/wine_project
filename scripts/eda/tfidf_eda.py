import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
import numpy as np
from scipy import sparse
from stopwords import *
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.manifold import TSNE
import cPickle as pickle
import seaborn


def read_files(names):

    for name in names:
        tmp = pd.read_csv(
            '../../data/dataframes/' + name)

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    return df


def sum_vectors_by_name(count_vecs, labels):
    n_unique = len(labels.unique())
    n_terms = count_vecs.shape[1]

    cv_by_name = np.zeros((n_unique, n_terms))

    for i, name in enumerate(labels.unique()):
        print i
        cv_by_name[i, :] = count_vecs[(labels == name).values, :].sum(axis=0)

    cv_by_name = sparse.csr_matrix(cv_by_name)

    return cv_by_name


def create_name_words(names):
    name_words = set()

    for name in names:
        words = map(lambda x: x.lower(), name.split())
        name_words.update(words)

    return name_words


def print_top_clust_words(clusts, cv_obj, tfidf_mat, n_words):

    keys = cv.vocabulary_.keys()
    vals = cv.vocabulary_.values()

    word_lookup = dict(zip(vals, keys))

    for i in np.unique(clusts):
        clust_cvs = tfidf_mat[clusts == i, :]
        ranked_words = np.fliplr(np.argsort(clust_cvs.mean(axis=0)))

        print "#################"
        print "CLUSTER NUMBER: ", i
        print "#################"

        for j in range(n_words):
            print word_lookup[ranked_words[0, j]]

        print '\n'


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
                         'brown']):

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
                        alpha=0.2)
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

if __name__ == "__main__":

    # read in data from cleaned csvs
    reviews = read_files(
        ['first_1000.csv', 'next_9000.csv', 'remainder_reviews.csv'])
    reviews = reviews[~reviews['review_text'].isnull()]

    # get text and name labels into numpy arrays
    X = reviews['review_text'].values
    y = reviews['wine_name']

    # unique name entries
    names = y.unique()

    # get unique words in the names, to strip from the count vectors
    name_words = create_name_words(names)

    # create stopwords set from all english stop words, plus words included in
    # wine names
    stopwords = name_words.union(english_stop_words)
    stopwords = stopwords.union(french_stop_words)
    stopwords = stopwords.union(wine_names_places)

    # taking out anything that occurs in a name was too brute force, remove
    # some common wine words from the stop list
    for w in wine_words:
        stopwords.remove(w)

    # instantiate count vectorizer
    cv = CountVectorizer(decode_error='ignore',
                         stop_words=stopwords,
                         ngram_range=(1, 1),
                         max_features=5000)

    # get count vector for each review
    count_vecs = cv.fit_transform(X)

    # sum count vectors for matching wine names
    cv_by_name = sum_vectors_by_name(count_vecs, labels=y)

    # tfidf transform the summed count vectors
    tf = TfidfTransformer()

    tfidf = tf.fit_transform(cv_by_name)

    # cluster using k-means
    # km = KMeans(n_clusters=2)

    # clusts = km.fit_predict(tfidf)

    # print out the top terms by cluster (avg magnitude of the tfidf vectors)
    # print_top_clust_words(clusts, cv, tfidf, 10)

    # latent semantic analysis (SVD on TFIDF)
    svd = TruncatedSVD(n_components=100)

    x_svd = svd.fit_transform(tfidf)

    # plot the X and Y principal components
    # plot_dim_red(x_svd, clusts, x=0, y=1)

    # NMF
    nmf = NMF(n_components=6)
    doc_matrix_nmf = nmf.fit_transform(tfidf)

    term_matrix_nmf = nmf.components_

    print_top_matrix_words(cv, term_matrix_nmf, 10)

    # pull out top topic from each point in nmf doc matrix
    nmf_top_topics = np.argsort(doc_matrix_nmf, axis=1)[:, -1]

    # # visualize with TSNE
    #
    # tsne = TSNE(n_components=2)
    #
    # x_tsne = tsne.fit_transform(x_svd)
    # plot_dim_red(x_tsne, clusts, x=0, y=1)

    # visualize top topics with tsne
    plot_dim_red(x_svd, nmf_top_topics, x=0, y=1)

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

    # visualize top topics with tsne
    plot_dim_red(x_svd, chablis, x=0, y=1)

    # Write pickle files to be loaded and used by predictor
    f = open('../predict/pickle_files/cv.pkl', 'w')
    pickle.dump(cv, f)
    f. close()

    f = open('../predict/pickle_files/tf.pkl', 'w')
    pickle.dump(tf, f)
    f. close()

    f = open('../predict/pickle_files/tfidf.pkl', 'w')
    pickle.dump(tfidf, f)
    f. close()
