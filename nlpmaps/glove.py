# import packages
import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# define functions for tokenizing the text data
class Word2VecVectorizer:
    def __init__(self, model_vec):
        print("Loading in word vectors...")
        self.word_vectors = model_vec
        print("Finished loading in word vectors")

    def fit(self, data):
        """fit data"""

    def transform(self, data):
        """determine the dimensionality of vectors"""
        v_get = self.word_vectors.get_vector('king')
        self.D = v_get.shape[0]

        x_vec = np.zeros((len(data), self.D))
        n_count = 0
        emptycount = 0

        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m_count = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m_count += 1
                except KeyError:
                    pass

            if len(vecs) > 0:
                vecs = np.array(vecs)
                x_vec[n_count] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n_count += 1

        print("Numer of samples with no words found: %s / %s" % (emptycount,
            len(data)))
        return x_vec

    def fit_transform(self, data):
        """transform the strings to vectors"""
        self.fit(data)

        return self.transform(data)

def glove_embedding(data, text_column: str, label_column: str):
    
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    file_name = 'glove.6B.zip'
    glove_file = 'glove.6B.100d.txt'
    word2vec_file = glove_file + '.word2vec'
    
    # Only download the file if it doesn't already exist
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)

    # Only extract the zip file if the glove file doesn't already exist
    if not os.path.exists(glove_file):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall()

    # Only convert the GloVe vectors to word2vec format if it hasn't already been done
    if not os.path.exists(word2vec_file):
        glove2word2vec(glove_file, word2vec_file)

    glove_model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    
    # set a word vectorizer
    vectorizer = Word2VecVectorizer(glove_model)
    
    text = data[text_column].values.reshape(-1,1)
    label = data[label_column].values.reshape(-1,1)
    
    text_list = text.tolist()
    text_str = [item for sublist in text_list for item in sublist]
    
    label_list = label.tolist()
    label_str = [item for sublist in label_list for item in sublist]

    # get the sentence embeddings for dataset
    embeddings = vectorizer.fit_transform(text_str)
    labels = label_str
    
    return embeddings
