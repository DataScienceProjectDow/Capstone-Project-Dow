import os
import scipy
import shutil
import tarfile
import unittest
import pandas as pds
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from ELMO_FINAL import *
from ELMO_FINAL import download_extract_imdb_data
from ELMO_FINAL import load_data
from unittest.mock import patch
from ELMO_FINAL import count_vectorizer


class TestDownloadExtractIMDBData(unittest.TestCase):

    def test_download_and_extract(self):
        # Test that the function downloads and extracts the IMDB data
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        filename = 'aclImdb_v1.tar.gz'
        extract_dir = 'aclImdb'
        expected_error = Exception

        try:
            root_folder = download_extract_imdb_data(url, filename, extract_dir)
        except Exception as e:
            self.assertIsInstance(e, expected_error)
        else:
            self.assertTrue(os.path.exists(root_folder))
            self.assertTrue(os.path.exists(os.path.join(root_folder, 'train')))
            self.assertTrue(os.path.exists(os.path.join(root_folder, 'test')))
            
    def tearDown(self):
        # Clean up the directory after each test
        root_folder = download_extract_imdb_data()
        shutil.rmtree(root_folder)



# class TestDataLoading(unittest.TestCase):
    def test_load_data(self):
        data = load_data("aclImdb/train")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 25000)


class TestCountVectorizer(unittest.TestCase):
    
    def test_count_vectorizer_with_list(self):
        x_train = [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'sentence']]
        x_test = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
        x_train_count, x_test_count = count_vectorizer(x_train, x_test)
        self.assertIsNotNone(x_train_count)
        self.assertIsNotNone(x_test_count)
    
    def test_count_vectorizer_with_nonlist(self):
        x_train = 'this is a sentence'
        x_test = 'this is another sentence'
        with self.assertRaises(TypeError):
            count_vectorizer(x_train, x_test)
            
            


class TestWordLevelTfIdf(unittest.TestCase):
    
    def test_word_level_tf_idf(self):
        x_train = [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'sentence']]
        x_test = [['this', 'is', 'yet', 'another', 'sentence'], ['and', 'this', 'is', 'a', 'third', 'one']]
        
        try:
            xtrain_tfidf, xtest_tfidf = word_level_tf_idf(x_train, x_test)
        except Exception as e:
            self.fail(f"word_level_tf_idf raised an exception: {str(e)}")
            
        self.assertIsInstance(xtrain_tfidf, scipy.sparse.csr.csr_matrix)
        self.assertIsInstance(xtest_tfidf, scipy.sparse.csr.csr_matrix)
        self.assertEqual(xtrain_tfidf.shape[0], len(x_train))
        self.assertEqual(xtest_tfidf.shape[0], len(x_test))

class TestNgramTfIdf(unittest.TestCase):
    
    def test_ngram_tf_idf(self):
        x_train = [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'sentence']]
        x_test = [['this', 'is', 'yet', 'another', 'sentence'], ['and', 'this', 'is', 'a', 'third', 'one']]
        
        try:
            xtrain_tfidf_ngram, xtest_tfidf_ngram = ngram_tf_idf(x_train, x_test)
        except Exception as e:
            self.fail(f"ngram_tf_idf raised an exception: {str(e)}")
        
        self.assertIsInstance(xtrain_tfidf_ngram, scipy.sparse.csr.csr_matrix)
        self.assertIsInstance(xtest_tfidf_ngram, scipy.sparse.csr.csr_matrix)
        self.assertEqual(xtrain_tfidf_ngram.shape[0], len(x_train))
        self.assertEqual(xtest_tfidf_ngram.shape[0], len(x_test))
        
class TestTrainModel(unittest.TestCase):
    
    def test_train_model(self):
        x_train = [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'sentence']]
        y_train = [0, 1]
        x_test = [['this', 'is', 'yet', 'another', 'sentence'], ['and', 'this', 'is', 'a', 'third', 'one']]
        y_test = [1, 0]
        
        # Test with a valid classifier and features
        rf = RandomForestClassifier()
        try:
            accuracy = train_model(rf, x_train, y_train, y_test, x_test)
        except Exception as e:
            self.fail(f"train_model raised an exception: {str(e)}")
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Test with an invalid classifier
        invalid_classifier = None
        with self.assertRaises(TypeError):
            accuracy = train_model(invalid_classifier, x_train, y_train, y_test, x_test)
        
        # Test with invalid feature vectors
        invalid_feature_vector = 'invalid feature vector'
        with self.assertRaises(TypeError):
            accuracy = train_model(rf, invalid_feature_vector, y_train, y_test, x_test)
        
        # Test with incompatible feature vectors and labels
        x_train_incompatible = [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'sentence'], ['this', 'is', 'yet', 'another', 'sentence']]
        with self.assertRaises(ValueError):
            accuracy = train_model(rf, x_train_incompatible, y_train, y_test, x_test)