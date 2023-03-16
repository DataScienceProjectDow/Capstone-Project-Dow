"""Unit test package for glove package"""

import unittest

import zipfile
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.ensemble

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from datasets import load_dataset

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

import sys

sys.path.insert(1, '..')

import glove

string = ['This is a movie review']

class TestWord2VecVectorizer(unittest.TestCase):
    """unit tests for functions in Word2VecVectorizer class"""

    def test_transform(self):
        self.assertEqual(len(glove.Word2VecVectorizer.transform(self, string)), 5)
        
    def test_fit_transform(self):
        self.assertEqual(len(glove.Word2VecVectorizer.fit_transform(self, string)[0], 100))
