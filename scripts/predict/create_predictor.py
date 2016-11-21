from gensim.models import Word2Vec
import cPickle as pickle

###
# import word2vec model
model = Word2Vec.load("../eda/wine_word2vec_model.model")

# take list of names and tfidf vector
# average up word vectors using tfidf weights to project each wine
# into word2vec space
f = open('pickle_files/cv.pkl', 'r')
cv = pickle.load(f)
f. close()

f = open('pickle_files/tf.pkl', 'r')
tf = pickle.load(f)
f. close()

f = open('pickle_files/tfidf.pkl', 'r')
tfidf = pickle.load(f)
f. close()


# if get varietal data, take clusters and calculate most typical
# varietal and country paring for each cluster

# predict!

# using list of words
# add them up, take cosine similarity across all wines

# to speed up maybe take cosine sim across clusters then only search
# within closest clusters?

# using wine in database, with option to add or subtract qualities
# create search tool for database
# add up vectors, taking cosine similarity across all wines

# optional: find nearest cluster and then suggest region-varietal pairs
# can do if I can scrap the data...
