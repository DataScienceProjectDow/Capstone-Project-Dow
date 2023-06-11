import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def preprocess_text(text, custom_words=None):
    """
    Preprocesses a text string by removing dates, times, punctuation,
    converting to lowercase, and removing stop words.

    Args:
        text (str): The text to preprocess.
        custom_words (set, optional): A set of custom words to remove in
            addition to the standard English stop words.

    Returns:
        str: The preprocessed text.
    """
    # Define stop words
    stop_words = set(stopwords.words('english'))
    if custom_words is not None:
        all_stop_words = stop_words.union(custom_words)
    else:
        all_stop_words = stop_words

    # Remove dates and times
    text = re.sub(
        r'(On\s[A-Z][a-z]+\s\d+(?:st|nd|rd|th)?\sat\s\d{1,2}:\d{2}\s(?:AM|PM))|(At\s\d{1,2}:\d{2}\s(?:AM|PM))',
        '', text)

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


def preprocessing(data, text_column, custom_words=None):
    """
    Applies the preprocess_text function to a specific column in a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the text data.
        text_column (str): The column in the DataFrame containing the text.
        custom_words (set, optional): A set of custom words to remove in
            addition to the standard English stop words.

    Returns:
        pd.DataFrame: The DataFrame with the preprocessed text in the specified column.
    """
    # Preprocess your text column
    data[text_column] = data[text_column].apply(
        lambda x: preprocess_text(x, custom_words=custom_words))

    return data
