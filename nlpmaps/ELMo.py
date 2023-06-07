import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def get_embeddings(data, text):
    elmo = hub.load('/Users/andrewsimon/Downloads/elmo_3')

    embeddings = []

    for sentence in data[text]:
        tensor = tf.constant([sentence])
        embedding = elmo.signatures['default'](tensor)['elmo']
        embedding = tf.reduce_mean(embedding, 1)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(len(embeddings), 1024)

    return embeddings