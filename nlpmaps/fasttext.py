import os
import gzip
import shutil
import wget
import numpy as np
from gensim.models.fasttext import load_facebook_vectors

class FastTextVectorizer:
    """Vectorizer using FastText model for transforming text data to vectors."""

    def __init__(self, fasttext_model):
        """
        Initializes FastTextVectorizer with a given FastText model.

        Args:
            fasttext_model: The pre-trained FastText model to be used for vector transformations.
        """
        self.fasttext_model = fasttext_model

    def transform(self, X):
        """
        Transforms the raw text data to vectors using the FastText model.

        The function iterates over words in each sample in X, checks if the word exists
        in the FastText model, gets its corresponding vector, and then averages all
        word vectors in each sample to obtain a single vector.

        Args:
            X: A list or array-like object of text data to be transformed.

        Returns:
            A numpy array where each row corresponds to the vector representation of
            each sample in X.
        """
        return np.array([
            np.mean([self.fasttext_model[w] for w in words.split() if w in self.fasttext_model]
                    or [np.zeros(self.fasttext_model.vector_size)], axis=0)
            for words in X
        ])


def fasttext_embedding(data, text_column: str):
    """
    Loads FastText model, transforms data to vectors, and returns these vectors.

    Args:
        data: A pandas DataFrame containing the data.
        text_column: The name of the column in 'data' that contains the text to be transformed.

    Returns:
        embeddings: A numpy array where each row is the vector representation of
        each text sample in the 'data'.
    """

    # Define the URL and the local path
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
    compressed_file_name = 'cc.en.300.bin.gz'
    uncompressed_file_name = 'cc.en.300.bin'

    # Only download the file if it doesn't already exist
    if not os.path.exists(compressed_file_name):
        wget.download(url, compressed_file_name)

    # Only uncompress the file if the uncompressed file doesn't already exist
    if not os.path.exists(uncompressed_file_name):
        with gzip.open(compressed_file_name, 'rb') as f_in:
            with open(uncompressed_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Load the vectors
    fasttext_model = load_facebook_vectors(uncompressed_file_name)

    text = data[text_column].values.reshape(-1, 1)

    text_list = text.tolist()
    text_str = [item for sublist in text_list for item in sublist]

    # Transform raw text data to vectors
    vectorizer = FastTextVectorizer(fasttext_model)
    embeddings = vectorizer.transform(text_str)

    return embeddings
