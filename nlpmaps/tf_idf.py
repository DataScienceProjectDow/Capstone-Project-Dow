import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_tfidf_embeddings(df, text_column, label_column):
    """
    Generate TF-IDF embeddings as arrays for each report from a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing text data and labels.
        text_column (str): The name of the column in df that contains the text data.
        label_column (str): The name of the column in df that contains the labels.
    
    Returns:
        pandas.DataFrame: A new DataFrame with the TF-IDF embeddings as arrays and labels.
    """
    text_data = df[text_column].values.tolist()
    labels = df[label_column].values
    
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer and transform the text data into TF-IDF embeddings
    embeddings = vectorizer.fit_transform(text_data)
    
    # Create a new DataFrame with embeddings as arrays and labels
    embeddings_df = pd.DataFrame(embeddings.toarray())
    embeddings_df[label_column] = labels
    
    return embeddings_df

