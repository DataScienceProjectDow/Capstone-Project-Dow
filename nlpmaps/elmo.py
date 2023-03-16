import glob
import os
import nltk
import gensim
import re
import spacy
import requests
import tarfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import pandas as pd
import xgboost


from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
parser = English()

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn import decomposition, ensemble
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import xgboost

import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
import tensorflow_hub as hub
import tensorflow as tf




nltk.download('stopwords')


def download_extract_imdb_data():
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filename = 'aclImdb_v1.tar.gz'
    extract_dir = 'aclImdb'
    
    # download file
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)

    # extract file
    tar = tarfile.open(filename, 'r:gz')
    tar.extractall()
    tar.close()
    
    # get path to root folder
    root_folder = os.path.join(os.getcwd(), extract_dir)
    
    return root_folder

def load_data(root):
    data = []
    filenames = ['neg' , 'pos']
    for sentiment in filenames:
        sentiment_dir = os.path.join(root, sentiment)
        for filename in os.listdir(sentiment_dir):
            with open(os.path.join(sentiment_dir, filename), 'r') as f:
                review = f.read()
                data.append((review, sentiment))
    return pd.DataFrame(data, columns=['review', 'sentiment'])



def preprocess_text(text):
    """
    a function to preprocess the text in a DataFrame column. 
    Takes in the review and sentiment columns of the train and test df
    """
    from gensim.utils import simple_preprocess
    # convert text to lowercase
    text = text.lower()
    # tokenize the text using gensim's simple_preprocess function
    tokens = simple_preprocess(text)
    # return the tokens as a string separated by spaces
    return ' '.join(tokens)


def count_vectorizer(x_train, x_test):
    """Count Vectors as features and creates a count vectorizer object""" 
    # Count Vectors as features
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit([' '.join(text) for text in x_train])
    count_vect.fit([' '.join(text) for text in x_test])

    # transform the training and test data using count vectorizer object
    xtrain_count =  count_vect.transform([' '.join(text) for text in x_train])
    xtest_count =  count_vect.transform([' '.join(text) for text in x_test])
    
    return xtrain_count, xtest_count

def word_level_tf_idf(x_train, x_test):
    """Word Level TF-IDF : Matrix representing tf-idf scores of every term in different documents"""

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit([' '.join(text) for text in x_train])
    tfidf_vect.fit([' '.join(text) for text in x_test])

    xtrain_tfidf =  tfidf_vect.transform([' '.join(text) for text in x_train])
    xtest_tfidf =  tfidf_vect.transform([' '.join(text) for text in x_test])
    return xtrain_tfidf, xtest_tfidf

def ngram_tf_idf(x_train, x_test):
    """ N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams"""    
    from nltk.corpus import stopwords
    stop_word = set(stopwords.words('english'))

    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000, stop_words='english')
    tfidf_vect_ngram.fit([' '.join(text) for text in x_train])
    tfidf_vect_ngram.fit([' '.join(text) for text in x_test])

    xtrain_tfidf_ngram = tfidf_vect_ngram.transform([' '.join(text) for text in x_train])
    xtest_tfidf_ngram = tfidf_vect_ngram.transform([' '.join(text) for text in x_test])
    
    return xtrain_tfidf_ngram,xtest_tfidf_ngram


def chara_level_tf_idf(x_train, x_test):
    """ Character Level TF-IDF : Matrix representing tf-idf scores of character level n-grams in the corpus"""
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit([' '.join(text) for text in x_train])
    tfidf_vect_ngram_chars.fit([' '.join(text) for text in x_test])

    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform([' '.join(text) for text in x_train]) 
    xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform([' '.join(text) for text in x_test]) 
    
    return xtrain_tfidf_chars,xtest_tfidf_chars


def hashing_vectorizer(x_train, x_test):
    hash_vectorizer = HashingVectorizer(n_features=5000)
    hash_vectorizer.fit([' '.join(text) for text in x_train])
    hash_vectorizer.fit([' '.join(text) for text in x_test])
    
    xtrain_hash_vectorizer =  hash_vectorizer.transform([' '.join(text) for text in x_train]) 
    xtest_hash_vectorizer =  hash_vectorizer.transform([' '.join(text) for text in x_test])
    return xtrain_hash_vectorizer, xtest_hash_vectorizer

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    """
    Model Building and Classification
    define classifier= Random Forest,Naive Bayes etc
    feature_vector_train=xtrain_tfidf/xtrain_tfidf_ngram/xtrain_tfidf_ngram-chars etc
    label= y_train
    feature_vector_valid=xtest_tfidf/xtest_tfidf_ngram/xtest_tfidf_ngram-chars etc
    """
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    return metrics.accuracy_score(predictions, y_test)


def ELMoEmbedding(x):
    """ELMO Embedding for input data after all vectorization"""
    with tf.compat.v1.variable_scope("my_scope", reuse=True):
        embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
    
    
def encode_labels(labels_train, labels_test):
    """
    Encodes a list of labels using one-hot encoding.

    Args:
    labels_train: A list of strings representing the training labels.
    labels_test: A list of strings representing the testing labels.

    Returns:
    A tuple of numpy arrays representing the one-hot encoded training and testing labels.
    """
    le = preprocessing.LabelEncoder()
    le.fit(labels_train + labels_test)

    def encode(le, labels):
        enc = le.transform(labels)
        return keras.utils.to_categorical(enc)

    labels_train_enc = encode(le, labels_train)
    labels_test_enc = encode(le, labels_test)
    
    return labels_train_enc, labels_test_enc


def build_model(x_train, y_train_enc):
    """
    Builds and trains a model using ELMo embeddings.
    
    Args:
    x_train: A list or numpy array of input strings.
    y_train_enc: A numpy array of one-hot-encoded labels.
    model_save_path: A string representing the file path to save the trained model weights.

    Returns:
    A trained keras model object.
    """
    x_train_elmo=[' '.join(text) for text in x_train]
    input_text = Input(shape=(1,), dtype=tf.string)
    with tf.compat.v1.variable_scope("my_scope_3",reuse=True):
        embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
        dense = Dense(256, activation='relu')(embedding)
        pred = Dense(2, activation='softmax')(dense)
        model = Model(inputs=[input_text], outputs=pred)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        K.get_session().run(tf.compat.v1.global_variables_initializer())
        K.get_session().run(tf.compat.v1.tables_initializer())
        history = model.fit(np.asarray(x_train_elmo), np.asarray(y_train_enc), epochs=5, batch_size=1000)
        model.save_weights('./elmo-model.h5')
        
    return model
