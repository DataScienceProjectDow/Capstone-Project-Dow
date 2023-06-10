import unittest
import pandas as pd
import sys

sys.path.insert(0, "..")

from data_preprocessing import preprocess_text, preprocessing

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.text = "On Monday 15th at 5:00 PM, there was an incident at the plant."
        self.custom_words = {"incident", "plant"}
        self.data = pd.DataFrame({"Report": ["On Monday 15th at 5:00 PM, there was an incident at the plant."]})

    def test_preprocess_text(self):
        result = preprocess_text(self.text, self.custom_words)
        self.assertIsInstance(result, str, "Result should be a string")
        self.assertNotIn("on monday 15th at 500 pm", result, "Dates and times should be removed")
        self.assertNotIn("there was an", result, "Stop words should be removed")

    def test_preprocessing(self):
        result = preprocessing(self.data, "Report", self.custom_words)
        self.assertIsInstance(result, pd.DataFrame, "Result should be a pandas DataFrame")
        self.assertIn("Report", result.columns, "DataFrame should contain 'Report' column")
        self.assertNotIn("on monday 15th at 500 pm", result["Report"].values[0], "Dates and times should be removed from DataFrame")
        self.assertNotIn("there was an", result["Report"].values[0], "Stop words should be removed from DataFrame")