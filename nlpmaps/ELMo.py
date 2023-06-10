import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def get_embeddings(data, text):
    """
    Inputs: data, pandas dataframe, dataframe containing
    text to be embedded by ELmO
            text, string, column name of text for embedding
    """

    # Load pretrained elmo model from tensorflow hub
    elmo = hub.load('https://tfhub.dev/google/elmo/3?tf-hub-format=compressed')

    # Intialize and empty array for embeddings
    embeddings = []

    # Go through each row in text column, calculating the
    # weighted average of each embedding
    for sentence in data[text]:
        tensor = tf.constant([sentence])
        embedding = elmo.signatures['default'](tensor)['elmo']
        embedding = tf.reduce_mean(embedding, 1)
        embeddings.append(embedding)

    # Reshape embeddings so that the array is passable to sci-kit learn
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(len(embeddings), 1024)

    return embeddings
