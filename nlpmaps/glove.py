# import packages
import os
import zipfile
import urllib.request
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


class Word2VecVectorizer:
    """
    A class to convert text data into word vectors using a trained model.
    """
    def __init__(self, model_vec):
        print("Loading in word vectors...")
        self.word_vectors = model_vec
        print("Finished loading in word vectors")

    def fit(self, data):
        """
        This function would be used if the model needed to learn something
        about the training data. In this case, it doesn't need to, so the
        function simply returns None.
        """

    def transform(self, data):
        """
        Convert a list of sentences into vectors. Unknown words are ignored.

        Args:
            data: A list of strings, where each string is a sentence.

        Returns:
            A 2D numpy array representing the sentence vectors.
        """
        # determine the dimensionality of vectors
        self.D = self.word_vectors.get_vector('king').shape[0]

        # prepare an empty numpy array to hold the sentence vectors
        x_vec = np.zeros((len(data), self.D))

        # iterate over the sentences in the data
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

        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        return x_vec

    def fit_transform(self, data):
        """
        A convenience function that performs fit and transform in one step.

        Args:
            data: A list of strings, where each string is a sentence.

        Returns:
            A 2D numpy array representing the sentence vectors.
        """
        self.fit(data)
        return self.transform(data)


def glove_embedding(data, text_column: str):
    """
    Download and extract GloVe vectors, convert them to word2vec format,
    and then use them to vectorize the given text data.

    Args:
        data: A pandas DataFrame containing the text data.
        text_column: The column in the DataFrame that contains the text data.
        label_column: The column in the DataFrame that contains the labels.

    Returns:
        embeddings: The sentence vectors obtained from the text data.
    """
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

    # Load the word2vec format GloVe vectors
    glove_model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

    # set a word vectorizer
    vectorizer = Word2VecVectorizer(glove_model)

    text = data[text_column].values.reshape(-1,1)

    # convert the nested list of text to a flat list of text
    text_str = [item for sublist in text.tolist() for item in sublist]

    # get the sentence embeddings for dataset
    embeddings = vectorizer.fit_transform(text_str)

    return embeddings
