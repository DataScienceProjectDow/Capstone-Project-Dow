import unittest
import sys
import pandas as pd

# Append the path where the module to test is located
sys.path.insert(0, "..")

from data_preprocessing import preprocess_text, preprocessing


class TestDataPreprocessing(unittest.TestCase):
    """
    Unit tests for the functions in the data_preprocessing module.
    """
    def setUp(self):
        """
        Define the test data that will be used in each test case.
        """
        self.text = "On Monday 15th at 5:00 PM, there was an incident at the plant."
        self.custom_words = {"incident", "plant"}
        self.data = pd.DataFrame(
            {"Report": ["On Monday 15th at 5:00 PM, there was an incident at the plant."]})

    def test_preprocess_text(self):
        """
        Test the preprocess_text function.
        """
        result = preprocess_text(self.text, self.custom_words)

        # Check that the result is a string
        self.assertIsInstance(result, str, "Result should be a string")

        # Check that dates and times have been removed
        self.assertNotIn("on monday 15th at 500 pm", result, "Dates and times should be removed")

        # Check that stop words have been removed
        self.assertNotIn("there was an", result, "Stop words should be removed")

    def test_preprocessing(self):
        """
        Test the preprocessing function.
        """
        result = preprocessing(self.data, "Report", self.custom_words)

        # Check that the result is a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame, "Result should be a pandas DataFrame")

        # Check that the DataFrame contains the 'Report' column
        self.assertIn("Report", result.columns, "DataFrame should contain 'Report' column")

        # Check that dates and times have been removed from the DataFrame
        self.assertNotIn(
            "on monday 15th at 500 pm", result["Report"].values[0],
            "Dates and times should be removed from DataFrame")

        # Check that stop words have been removed from the DataFrame
        self.assertNotIn(
            "there was an", result["Report"].values[0],
            "Stop words should be removed from DataFrame")
