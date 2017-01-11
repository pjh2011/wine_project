import numpy as np
from plotting_TSNE_and_hists import read_files, load_files
from scipy.spatial.distance import cosine
import pandas as pd
import string


def search_vecs(vector, vectors, names):
    dists = np.apply_along_axis(lambda x: cosine(x, vector),
                                axis=1,
                                arr=vectors)
    topMatchesInd = np.argsort(dists)

    print names[topMatchesInd][0:5]
    print dists[topMatchesInd][0:5]


def clean_accents(s):
    # ChAC/teau LA(c)oville Barton
    # Chateau Leoville Barton
    # A1/4 = u
    # MoA<<t = Moet
    replace_dict = {'AC/': 'a', 'A(c)': 'e', 'A1/4': 'u',
                    'A<<': 'e', '(r).': 'r', 'A"': 'e', 'A+': 'n',
                    'A3': 'o', 'A(r)': 'i', 'A-': 'i', 'A!': 'a'}

    for accent in replace_dict:
        s = string.replace(s, accent, replace_dict[accent])

    return s

if __name__ == "__main__":

    model, cv, tf, tfidf, names, wineVectors = load_files()

    non_zero = np.where(tfidf.sum(axis=1) != 0)[0]
    tfidf = tfidf[non_zero, :]
    names = names[non_zero]

    names = np.array(map(clean_accents, names))

    weights = tfidf.max(axis=1).todense()

    words = zip(model.vocab.keys(), model.vocab.values())
    words = map(lambda x: (x[0], x[1].index), words)
    words = sorted(words, key=lambda x: x[1])
    words = np.array(words)[:, 0]

    wordVectors = model.syn0

    search_vecs(wordVectors[571], wordVectors, words)

    # dump to raw data for R
    np.savetxt("../../shiny_app/data/wineVectors.csv", wineVectors,
               delimiter=",", fmt="%s")
    np.savetxt("../../shiny_app/data/weights.csv", weights,
               delimiter=",", fmt="%s")

    np.savetxt("../../shiny_app/data/wordVectors.csv", wordVectors,
               delimiter=",", fmt="%s")

    np.savetxt("../../shiny_app/data/names.txt", names,
               delimiter="\n", fmt="%s")
    np.savetxt("../../shiny_app/data/words.txt", words,
               delimiter="\n", fmt="%s")
