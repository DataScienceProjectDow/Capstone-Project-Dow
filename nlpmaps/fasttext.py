from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import load_facebook_vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn
import optuna
import os

class FastTextVectorizer:
    def __init__(self, fasttext_model):
        self.fasttext_model = fasttext_model

    def transform(self, X):
        return np.array([
            np.mean([self.fasttext_model[w] for w in words.split() if w in self.fasttext_model]
                    or [np.zeros(self.fasttext_model.vector_size)], axis=0)
            for words in X
        ])

def fasttext_embedding(data, text_column: str, label_column: str):
    
    # Define the URL and the local path
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
    compressed_file_name = 'cc.en.300.bin.gz'
    uncompressed_file_name = 'cc.en.300.bin'

    # Only download the file if it doesn't already exist
    if not os.path.exists(compressed_file_name):
        import wget
        wget.download(url, compressed_file_name)

    # Only uncompress the file if the uncompressed file doesn't already exist
    if not os.path.exists(uncompressed_file_name):
        import gzip
        import shutil
        
        with gzip.open(compressed_file_name, 'rb') as f_in:
            with open(uncompressed_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Load the vectors
    fasttext_model = load_facebook_vectors(uncompressed_file_name)
    
    text = data[text_column].values.reshape(-1,1)
    label = data[label_column].values.reshape(-1,1)
    
    text_list = text.tolist()
    text_str = [item for sublist in text_list for item in sublist]
    
    label_list = label.tolist()
    label_str = [item for sublist in label_list for item in sublist]
    
    # Transform raw text data to vectors
    vectorizer = FastTextVectorizer(fasttext_model)
    embeddings = vectorizer.transform(text_str)
    
    return pd.DataFrame({'Embeddings': embeddings.tolist(), 'Labels': label_str})