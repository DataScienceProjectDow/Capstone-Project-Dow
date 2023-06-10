# Library Imports
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader

tqdm.pandas(desc="progress-bar")
tokenizer = TweetTokenizer()
pd.options.mode.chained_assignment = None


def tokenize(report):
    """
    Inputs: report, string, report to be tokenized

    Helper function for the tokenizer, tokenizes reports uses NTLK Tokenizer
    and return "NC" if OOV. Return tokenized entry
    """
    # Try to tokenize report, if not return "NC"
    try:
        tokens = tokenizer.tokenize(report)
        return tokens
    except:
        return "NC"


def postprocess(data, label):
    """
    Inputs: data, pandas dataframe, dataframe containing data to be embedded
    by W2v
            label string, string of column name of labels

    Wrapped tokenizer function that tokenizes given text in pandas dataframe.
    Return dataframe with text column labelized
    """

    # Take tokenized text and append column to dataframe of tokens
    data['tokens'] = data[label].progress_map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


def labelizeReports(reports, label_type):
    """
    Inputs: reports, series, list of reports that have been tokenized
            label_type, string, label type of data i.e train, test , etc

    Uses Doc2Vec labelizer to tag  entries in pandas dataframe,
    useful for machine learning Return labelized text
    """
    # Empty array for labelizing
    labelized = []

    # Iterate through reports and append tagged labelization
    for i, v in tqdm(enumerate(reports)):
        label = '%s_%s' % (label_type, i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized


def extract(vecs):
    """
    Inputs: vecs, np array, array of vectors from labeled, tokenized reports

    Builds TFIDF matrix for Word2Vec model, returns TFIDF matrix
    """
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in vecs])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    return tfidf


# Load pretrained google vectors into global variable for script
google_vectors = gensim.downloader.load('word2vec-google-news-300')


def buildWordVector(tokens, size, tfidf):
    """
    Inputs: tokens,tokens to be extracted from W2V model
            size, embedding dimension size (this is locked to
            300 by google vecs)
            tfidf, tfidf matrix used for embedding

    Builds word vectors and averages each embedding for classification. Returns
    a vector embedded based on a single token. Return "size dimensional vector
    """

    # Iterate words in token and build averaged word vector for sentence.
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += google_vectors[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count

    return vec


def return_embeddings(vecs, tfidf):
    """
    Inputs, vecs, vectorized tokens from buildWordVector function
            tfidf, tfidf matrix for vectorization

    creates embedded array of word vectors by iterating
    through each token in data and vectorizing
    returns np array of embedded vectors
    """

    # Iterate through data and build word vectors by calling buildWordVector
    return np.concatenate([buildWordVector(z, 300, tfidf) for z in tqdm(map(lambda x: x.words, vecs))])


def get_embeddings(data, text):
    """
    Inputs: data, pandas dataframe, pandas dataframe containing
            text to be vectorized
            text, string, colummn name of text to be vectorized

    Main function of script. Receives pandas dataframe with text
    to be vectorized, return numpy array of embededed text vectors.
    """

    # Call functions
    data = postprocess(data, text)
    vecs = labelizeReports(data.tokens, 'all')
    tfidf = extract(vecs)
    embeddings = return_embeddings(vecs, tfidf)
    return embeddings
