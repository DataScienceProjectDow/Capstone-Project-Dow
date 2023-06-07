# Library Imports

import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.

import gensim

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
from gensim.models.doc2vec import TaggedDocument



from sklearn.feature_extraction.text import TfidfVectorizer

import gensim.downloader

# Tokenizer helper function
def tokenize(report):
    try:
        tokens = tokenizer.tokenize(report)
        return tokens
    except:
        return "NC"
    
# Tokenizer wrapping function
def postprocess(data, label):
    data['tokens'] = data[label].progress_map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

# Attach Doc2vec labels to each of our tokens
def labelizeReports(reports, label_type):
    labelized = []
    for i,v in tqdm(enumerate(reports)):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized


def extract(vecs):
     vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
     matrix = vectorizer.fit_transform([x.words for x in vecs])
     tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
     return tfidf

google_vectors = gensim.downloader.load('word2vec-google-news-300')

def buildWordVector(tokens, size, tfidf):

    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += google_vectors[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count

    return vec

def return_embeddings(vecs, tfidf):
    return np.concatenate([buildWordVector(z, 300, tfidf) for z in tqdm(map(lambda x: x.words, vecs))])

def get_embeddings(data, text):
    data = postprocess(data, text)
    vecs = labelizeReports(data.tokens, 'all')
    tfidf = extract(vecs)
    embeddings = return_embeddings(vecs, tfidf)
    return embeddings