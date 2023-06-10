import unittest
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bag_of_words import generate_bow_embeddings

class GenerateBowEmbeddingsTest(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        self.df = pd.DataFrame({
            'text': ['This is sentence 1', 'Another sentence'],
            'label': [0, 1]
        })
        
        self.text_column = 'text'
        self.label_column = 'label'
        
        # Create a CountVectorizer
        self.vectorizer = CountVectorizer()
        
    def test_generate_bow_embeddings(self):
        # Call the function to generate embeddings and labels
        embeddings_df = generate_bow_embeddings(self.df, self.text_column, self.label_column)
        
        # Assertions
        self.assertIsInstance(embeddings_df, pd.DataFrame)
        self.assertEqual(embeddings_df.shape[0], self.df.shape[0])
        self.assertEqual(set(embeddings_df.columns),
                         set(self.vectorizer.get_feature_names_out() + [self.label_column]))
        
