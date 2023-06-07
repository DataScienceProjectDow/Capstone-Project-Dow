import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def generate_bow_embeddings(df, text_column, label_column):
    """
    Generate bag-of-words embeddings and labels from a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing text data and labels.
        text_column (str): The name of the column in df that contains the text data.
        label_column (str): The name of the column in df that contains the labels.
    
    Returns:
        tuple: A tuple containing the bag-of-words embeddings and labels.
            - embeddings (scipy.sparse.csr_matrix): The bag-of-words embeddings of the text data.
            - labels (numpy.ndarray): The labels corresponding to the text data.
    """
    text_data = df[text_column].values.tolist()
    labels = df[label_column].values
    
    # Create a CountVectorizer
    vectorizer = CountVectorizer()
    
    # Fit the vectorizer on the text data and transform the data into bag-of-words embeddings
    embeddings = vectorizer.fit_transform(text_data)
    
    return embeddings, labels
