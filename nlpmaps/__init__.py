__version__ = "1.0.0"
from bag_of_words import generate_bow_embeddings
from tf_idf import generate_tfidf_embeddings
from Word2Vec import get_embeddings as get_word2vec_embeddings
from glove import glove_embedding
from fasttext import fasttext_embedding
from ELMo import get_embeddings as get_elmo_embeddings
from bert import generate_bert_emebeddings
from selection import find_optimal_method