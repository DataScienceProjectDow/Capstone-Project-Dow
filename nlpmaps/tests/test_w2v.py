import pandas as pd
import unittest
import sys

sys.path.append('/Users/andrewsimon/nlpmaps/nlpmaps')

# Note: this throws a flake8 error however is required by sys
# in order to import word2vec
import Word2Vec


class Test_df_utils(unittest.TestCase):

    def test_get_embeddings(self):
        """Testing that Word2vec returns proper dimension arrays"""

        # Import data
        main_data = pd.read_csv('/Users/andrewsimon/Desktop/Dow_dat.csv')
        embeddings = Word2Vec.get_embeddings(main_data, "Report")

        # Asserting that word2vec is returning arrays of proper dimensionality
        assert len(embeddings) == 300
        assert len(embeddings[0]) == 300
