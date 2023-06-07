import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.ensemble
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('talk')

def read_excel_file(file_path):
    """
    Read an Excel file and combine all sheets into a single DataFrame.
    
    Args:
        file_path (str): The path to the Excel file.
    
    Returns:
        pandas.DataFrame: The combined DataFrame containing data from all sheets.
    """
    excel_data = pd.read_excel(file_path, sheet_name=None)
    all_sheets = list(excel_data.values())
    combined_data = pd.concat(all_sheets, ignore_index=True)
    return combined_data

def split_train_test_data(data, test_size=0.2, random_state=206):
    """
    Split the data into training and testing sets.
    
    Args:
        data (pandas.DataFrame): The input data to be split.
        test_size (float): The proportion of the data to be used for testing.
        random_state (int): The random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing the training set and testing set.
    """
    train, test = sklearn.model_selection.train_test_split(data, test_size=test_size, random_state=random_state)
    return train, test

def tokenize_text(train_text, test_text):
    """
    Tokenize the text data using CountVectorizer.
    
    Args:
        train_text (numpy.ndarray): The text data for training.
        test_text (numpy.ndarray): The text data for testing.
    
    Returns:
        tuple: A tuple containing the tokenized training data and tokenized testing data.
    """
    vectorizer = CountVectorizer()
    train_x = vectorizer.fit_transform(train_text)
    test_x = vectorizer.transform(test_text)
    return train_x, test_x

def train_random_forest(train_x, train_y):
    """
    Train a Random Forest Classifier and perform hyperparameter tuning.
    
    Args:
        train_x (scipy.sparse.csr_matrix): The tokenized training data.
        train_y (numpy.ndarray): The target labels for training.
    
    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained Random Forest model.
    """
    rf_model = RandomForestClassifier(n_estimators=200, random_state=206)
    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = sklearn.model_selection.GridSearchCV(rf_model, params, cv=5)
    grid_search.fit(train_x, train_y)
    best_rf_model = grid_search.best_estimator_
    return best_rf_model

def train_classification_models(train_x, train_y):
    """
    Train SVM and Naive Bayes models.
    
    Args:
        train_x (scipy.sparse.csr_matrix): The tokenized training data.
        train_y (numpy.ndarray): The target labels for training.
    
    Returns:
        tuple: A tuple containing the trained SVM model and Naive Bayes model.
    """
    svm_model = SVC()
    nb_model = MultinomialNB()
    svm_model.fit(train_x, train_y)
    nb_model.fit(train_x, train_y)
    return svm_model, nb_model

def train_voting_classifier(best_rf_model, svm_model, nb_model, train_x, train_y):
    """
    Train a Voting Classifier using the best Random Forest, SVM, and Naive Bayes models.
    
    Args:
        best_rf_model (sklearn.ensemble.RandomForestClassifier): The best Random Forest model.
        svm_model (sklearn.svm.SVC): The trained SVM model.
        nb_model (sklearn.naive_bayes.MultinomialNB): The trained Naive Bayes model.
        train_x (scipy.sparse.csr_matrix): The tokenized training data.
        train_y (numpy.ndarray): The target labels for training.
    
    Returns:
        sklearn.ensemble.VotingClassifier: The trained Voting Classifier model.
    """
    voting_model = VotingClassifier(
        estimators=[('rf', best_rf_model), ('svm', svm_model), ('nb', nb_model)],
        voting='hard'
    )
    voting_model.fit(train_x, train_y)
    return voting_model

def evaluate_models(test_x, test_y, models):
    """
    Evaluate the accuracy of multiple models on the test data and generate classification reports.
    
    Args:
        test_x (scipy.sparse.csr_matrix): The tokenized testing data.
        test_y (numpy.ndarray): The target labels for testing.
        models (list): A list of trained models to be evaluated.
    
    Returns:
        tuple: A tuple containing the accuracy scores and classification reports for each model.
    """
    accuracy_scores = []
    classification_reports = []
    for model in models:
        accuracy = model.score(test_x, test_y)
        accuracy_scores.append(accuracy)
        predictions = model.predict(test_x)
        report = sklearn.metrics.classification_report(test_y, predictions, digits=4)
        classification_reports.append(report)
    return accuracy_scores, classification_reports

def plot_confusion_matrix(test_y, predictions):
    """
    Plot the confusion matrix based on the true labels and predicted labels.
    
    Args:
        test_y (numpy.ndarray): The true labels.
        predictions (numpy.ndarray): The predicted labels.
    """
    cm = sklearn.metrics.confusion_matrix(test_y, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
