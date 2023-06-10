import pandas as pd
import unittest
import sys

sys.path.append('/Users/andrewsimon/nlpmaps/nlpmaps')

# Note: this throws a flake8 error however is required by sys
# in order to import word2vec
import selection


class Test_df_utils(unittest.TestCase):

    def test_find_optimal_method(self):
        """
        Testing that ELMo selection algorithm
        returns proper dimension datframe
        """

        # Import data
        main_data = pd.read_csv('/Users/andrewsimon/Desktop/Dow_dat.csv')

        # Run selection algorithm
        methods = selection.find_optimal_method(main_data, "Report")

        # Asserting that selection algorithm is
        # returning a pd with all 28 scores
        assert len(methods) == 4
        assert len(methods.columns) == 7
