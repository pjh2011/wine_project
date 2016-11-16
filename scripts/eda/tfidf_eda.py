import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
import numpy as np
from scipy import sparse
from stopwords import english_stop_words


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


def print_top_words(clusts, cv_obj, tfidf_mat, n_words):

    keys = cv.vocabulary_.keys()
    vals = cv.vocabulary_.values()

    word_lookup = dict(zip(vals, keys))

    for j in np.unique(clusts):
        clust_cvs = tfidf_mat[clusts == j, :]
        ranked_words = np.argsort(clust_cvs.mean(axis=0))

        print "#################"
        print "CLUSTER NUMBER: ", j
        print "#################"

        for i in range(1, n_words + 1):
            print word_lookup[ranked_words[0, -i]]

        print '\n'

if __name__ == "__main__":

    # read in data from cleaned csvs
    reviews = read_files(['first_1000.csv', 'next_9000.csv'])
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

    # instantiate count vectorizer
    cv = CountVectorizer(decode_error='ignore', stop_words=stopwords)

    # get count vector for each review
    count_vecs = cv.fit_transform(X)

    # sum count vectors for matching wine names
    cv_by_name = sum_vectors_by_name(count_vecs, labels=y)

    tf = TfidfTransformer()

    tfidf = tf.fit_transform(cv_by_name)

    km = KMeans()

    clusts = km.fit_predict(tfidf)
