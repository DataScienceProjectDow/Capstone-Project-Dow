import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import sys

import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "..")

from glove import Word2VecVectorizer, glove_embedding

class TestWord2VecVectorizer(unittest.TestCase):
    def setUp(self):
        self.model_vec = MagicMock()  # Mock the model vectors
        self.model_vec.get_vector.return_value = np.zeros(100)  # Assume 100D vectors
        self.vectorizer = Word2VecVectorizer(self.model_vec)

    def test_transform(self):
        data = ['This is a test sentence.', 'This is another test sentence.']
        transformed_data = self.vectorizer.transform(data)

        # Check the shape of the returned data
        self.assertEqual(transformed_data.shape[0], len(data))
        self.assertEqual(transformed_data.shape[1], 100)  # The dimensionality we set

        # Check that the returned data is a numpy array
        self.assertIsInstance(transformed_data, np.ndarray)

    def test_fit_transform(self):
        data = ['This is a test sentence.', 'This is another test sentence.']
        
        # Mock the fit and transform methods
        with patch.object(self.vectorizer, 'fit') as mock_fit, patch.object(self.vectorizer, 'transform') as mock_transform:
            mock_transform.return_value = 'transformed_data'
            result = self.vectorizer.fit_transform(data)

        # Check that fit and transform were called once
        mock_fit.assert_called_once_with(data)
        mock_transform.assert_called_once_with(data)

        # Check that the result of fit_transform is the result from the transform method
        self.assertEqual(result, 'transformed_data')

class TestGloveEmbedding(unittest.TestCase):
    @patch('os.path.exists')
    @patch('pandas.read_excel')
    @patch('zipfile.ZipFile')
    @patch('urllib.request.urlretrieve')
    @patch('glove.Word2VecVectorizer')
    @patch('gensim.models.KeyedVectors.load_word2vec_format')
    
    def test_glove_embedding(
        self, mock_load_word2vec_format, mock_Word2VecVectorizer,
        mock_urlretrieve, mock_ZipFile, mock_read_excel, mock_exists):
        
        # Mock the return values of the functions we're not testing
        mock_exists.return_value = False
        mock_read_excel.return_value = pd.DataFrame(
            {'text': ['This is a test sentence.', 'This is another test sentence.'], 'label': [1, 0]})
        mock_load_word2vec_format.return_value = MagicMock()  # Mock the word2vec model
        mock_Word2VecVectorizer.return_value = MagicMock()  # Mock the vectorizer
        mock_Word2VecVectorizer.return_value.fit_transform.return_value = np.zeros((2, 100))  # Assume 2 sentences, 100D vectors

        path = 'dummy_path'
        text_column = 'text'
        label_column = 'label'
        result = glove_embedding(path, text_column, label_column)

        # Check the shape of the returned data
        self.assertEqual(result.shape[0], 2)  # The number of sentences we set
        self.assertEqual(result.shape[1], 2)  # The number of columns in the dataframe ('Embeddings' and 'Labels')

        # Check that the returned data is a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)