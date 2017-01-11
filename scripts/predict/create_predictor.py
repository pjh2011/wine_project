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

    f = open('pickle_files/wine_vectors.pkl', 'w')
    pickle.dump(wine_vectors, f)
    f. close()

    # predict!
    j = 4000
    wv = wine_vectors[j, :]  # search vector
    dists = np.apply_along_axis(lambda x: cosine(x, wv),
                                axis=1,
                                arr=wine_vectors)

    print names[np.argsort(dists)][1:5]

    w = tfidf[j, :].todense().max()

    wv1 = wv + w * model['white'] - w * model['red']
    dists = np.apply_along_axis(lambda x: cosine(x, wv1),
                                axis=1,
                                arr=wine_vectors)
    print names[np.argsort(dists)][1:5]

    # using list of words
    # add them up, take cosine similarity across all wines

    wv2 = model['butter'] + model['citrus'] + model['golden']
    dists = np.apply_along_axis(lambda x: cosine(x, wv2),
                                axis=1,
                                arr=wine_vectors)
    print names[np.argsort(dists)][1:5]

    # https://github.com/seatgeek/fuzzywuzzy to do fuzzy term matching?
