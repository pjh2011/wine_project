import pandas as pd
from stopwords import *
import nltk.data
from nltk.tokenize import word_tokenize
from tfidf_eda import create_name_words
from gensim.models import word2vec

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def read_files(names):

    for name in names:
        tmp = pd.read_csv(
            '../../data/dataframes/' + name)

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    return df


def review_to_sentences(review, sent_tokenizer):

    raw_sentences = sent_tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            sentences.append(map(lambda x: x.lower(),
                                 word_tokenize(raw_sentence)))

    return sentences

reviews = read_files(['first_1000.csv', 'next_9000.csv'])
reviews = reviews[~reviews['review_text'].isnull()]

y = reviews['wine_name']

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set "
print str(len(reviews['review_text'])) + ' total reviews'

for i, review in enumerate(reviews["review_text"]):
    if i % 100 == 0:
        print i
    sentences += review_to_sentences(review, tokenizer)


# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Set values for various parameters
num_features = 1000    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 8       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "wine_word2vec_model"
model.save(model_name)

print model.most_similar("corked")
print model.most_similar("fruit")
