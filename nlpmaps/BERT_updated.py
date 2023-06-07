import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

def generate_bert_embeddings(df, text_column, label_column):
    """
    Generate BERT embeddings and labels from a DataFrame using the bert-base-uncased model.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing text data.
        text_column (str): The name of the column in df that contains the text data.
        label_column (str): The name of the column in df that contains the labels.
    
    Returns:
        pandas.DataFrame: A new DataFrame containing the BERT embeddings and labels.
            The new DataFrame has two columns:
            - 'Embeddings': The BERT embeddings of the text data.
            - 'Labels': The labels corresponding to the text data.
    """
    text_data = df[text_column].values.tolist()
    labels = df[label_column].values
    
    # Load the BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Tokenize the text data
    encoded_inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
    
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        embeddings = outputs.last_hidden_state
    
    # Create a new DataFrame with embeddings and labels
    new_df = pd.DataFrame({
        'Embeddings': embeddings.tolist(),
        'Labels': labels
    })
    
    return new_df
