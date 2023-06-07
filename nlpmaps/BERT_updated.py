import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

def generate_bert_embeddings(df, text_column, label_column):
    """
    Generate BERT embeddings and labels from a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing text data and labels.
        text_column (str): The name of the column in df that contains the text data.
        label_column (str): The name of the column in df that contains the labels.
    
    Returns:
        tuple: A tuple containing the BERT embeddings and labels.
            - embeddings (torch.Tensor): The BERT embeddings of the text data.
            - labels (numpy.ndarray): The labels corresponding to the text data.
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
    
    return embeddings, labels
