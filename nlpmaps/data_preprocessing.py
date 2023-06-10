import nltk
import string
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text, custom_words=None):
    
    # Define stop words
    stop_words = set(stopwords.words('english'))
    if custom_words is not None:
        all_stop_words = stop_words.union(custom_words)
    else:
        all_stop_words = stop_words

    # Remove dates and times
    text = re.sub(r'(On\s[A-Z][a-z]+\s\d+(?:st|nd|rd|th)?\sat\s\d{1,2}:\d{2}\s(?:AM|PM))|(At\s\d{1,2}:\d{2}\s(?:AM|PM))', '', text)

    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words and custom words
    words = [word for word in words if word not in all_stop_words]

    # Join the words back into a single string and return it
    return " ".join(words)

def preprocessing(data, text_column: str, custom_words=None):
    
    # Preprocess your text column
    data[text_column] = data[text_column].apply(preprocess_text, custom_words=custom_words)
    
    return data