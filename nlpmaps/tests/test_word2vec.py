import pandas as pd
import unittest
import sys
import math

sys.path.append('/Users/andrewsimon/nlpmaps/nlpmaps')

# Note: this throws a flake8 error however is required by sys
# in order to import word2vec
import word2vec


class Test_df_utils(unittest.TestCase):

    def test_to_pd_dataframe(self):
        """Testing that to_pd_dataframe returns a df"""
        df = word2vec.to_pd_dataframe(
            '/Users/andrewsimon/Desktop/IMDBDataset.csv.zip')

        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_pre_process(self):
        """Testing that preprocessing function cleans df"""

        df = pd.read_csv('/Users/andrewsimon/Desktop/IMDBDataset.csv.zip')

        df_process = word2vec.pre_process(df, 'review', 'sentiment',
                                          'positive', 'negative')

        assert df_process.at[0, 'sentiment'] == 1

    def test_sample_data(self):
        """Testing that data sampler return expected length"""

        df = pd.read_csv('/Users/andrewsimon/Desktop/IMDBDataset.csv.zip')
        sampled_data = word2vec.sample_data(df, 10)

        assert len(sampled_data) == 10

    def test_data_split(self):
        """Testing that test fraction is equal to actual split in the data"""
        df = pd.read_csv('/Users/andrewsimon/Desktop/IMDBDataset.csv.zip')
        df_process = word2vec.pre_process(df, 'review', 'sentiment',
                                          'positive', 'negative')

        testing_size = 0.2

        X_train, X_test, y_train, y_test = word2vec.data_split(df_process,
                                                               'review',
                                                               'sentiment',
                                                               testing_size)

        test_fraction = len(X_test) / len(df_process)

        self.assertTrue(math.isclose(test_fraction, testing_size))

    def test_vectorize(self):
        """Test that vector proportions are conserved in vectorization"""
        df = pd.read_csv('/Users/andrewsimon/Desktop/IMDBDataset.csv.zip')

        df_process = word2vec.pre_process(df, 'review', 'sentiment',
                                          'positive', 'negative')

        df_sampled = word2vec.sample_data(df_process, 10)

        testing_size = 0.2

        X_train, X_test, y_train, y_test = word2vec.data_split(df_sampled,
                                                               'review',
                                                               'sentiment',
                                                               testing_size)

        X_train_vect, X_test_vect = word2vec.vectorize(X_train, X_test)

        test_fraction = len(X_test_vect) / (len(X_test_vect) +
                                            len(X_train_vect))

        self.assertTrue(math.isclose(test_fraction, testing_size))

    def test_make_prediction(self):
        """Test that the accuracy score returns a float"""
        df = pd.read_csv('/Users/andrewsimon/Desktop/IMDBDataset.csv.zip')

        df_process = word2vec.pre_process(df, 'review', 'sentiment',
                                          'positive', 'negative')

        df_sampled = word2vec.sample_data(df_process, 10)

        testing_size = 0.2

        X_train, X_test, y_train, y_test = word2vec.data_split(df_sampled,
                                                               'review',
                                                               'sentiment',
                                                               testing_size)

        X_train_vect, X_test_vect = word2vec.vectorize(X_train, X_test)

        acc_score = word2vec.make_prediction(X_train_vect, X_test_vect,
                                             y_train, y_test)

        self.assertTrue(isinstance(acc_score, float))

    def test_accuracy_prediction(self):
        """Test that the accuracy returned by function wrappers is also
        a floating point number"""

        acc_score = word2vec.accuracy_prediction(
            '/Users/andrewsimon/Desktop/IMDBDataset.csv.zip',
            'review', 'sentiment', 'positive', 'negative',
            10)

        self.assertTrue(isinstance(acc_score, float))
