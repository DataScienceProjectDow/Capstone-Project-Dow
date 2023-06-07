import unittest
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from bert import generate_bert_embeddings

class GenerateBertEmbeddingsTest(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        self.df = pd.DataFrame({
            'text': ['This is sentence 1', 'Another sentence'],
            'label': [0, 1]
        })
        
        self.text_column = 'text'
        self.label_column = 'label'
        
        # Load the BERT model and tokenizer
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        
    def test_generate_bert_embeddings(self):
        # Call the function to generate embeddings and labels
        embeddings, labels = generate_bert_embeddings(self.df, self.text_column, self.label_column,
                                                      self.tokenizer, self.model)
        
        # Assertions
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(embeddings.shape[0], self.df.shape[0])
        self.assertEqual(labels.shape[0], self.df.shape[0])
        # Add more assertions as needed
        
if __name__ == '__main__':
    unittest.main()
