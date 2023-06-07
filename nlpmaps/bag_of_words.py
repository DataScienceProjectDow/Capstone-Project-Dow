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
        pandas.DataFrame: A new DataFrame with the bag-of-words embeddings and labels.
    """
    text_data = df[text_column].values.tolist()
    labels = df[label_column].values
    
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()
    
    # Fit the vectorizer and transform the text data into bag-of-words embeddings
    embeddings = vectorizer.fit_transform(text_data)
    
    # Create a new DataFrame with embeddings and labels
    embeddings_df = pd.DataFrame(embeddings.toarray())
    embeddings_df.columns = vectorizer.get_feature_names_out()
    embeddings_df[label_column] = labels
    
    return embeddings_df
