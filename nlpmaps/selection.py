# Import Models
import pandas as pd
from sklearn.model_selection import train_test_split

import tf_idf
import ELMo
import bag_of_words
import BERT_updated
import fasttext
import wget
import glove
import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def get_tfidf_embeddings(data, text, labels):
    """
    Inputs: data: pandas dataframe of text for embedding with labels
            text: string, column name of text to be embedded
            labels: string, column name of text labels

    Extracts tfidf embeddings of text in a pandas dataframe, returns array of
    embedded vectors replacing text column
    """

    # Extract and return tf_idf embeddings
    return tf_idf.generate_tfidf_embeddings(data, text, labels).drop(columns=labels).values


def get_BoW_embeddings(data, text, labels):
    """
    Inputs: data: pandas dataframe of text for embedding with labels
            text: string, column name of text to be embedded
            labels: string, column name of text labels

    Extracts Bag of Words  embeddings of text in a
    pandas dataframe, returns array of
    embedded vectors replacing text column
    """
    # Extract and return bag of words embeddings
    return bag_of_words.generate_bow_embeddings(data, text, labels).drop(columns=labels).values


def get_bert_embeddings(data, text, labels):
    """
    Inputs: data: pandas dataframe of text for embedding with labels
            text: string, column name of text to be embedded
            labels: string, column name of text labels

    Extracts BERT embeddings of text in a pandas dataframe, returns array of
    embedded vectors replacing text column
    """
    # Extract and return bert embeddings
    return BERT_updated.generate_bert_embeddings(data, text, labels)


def get_Word2Vec_embeddings(data, text):
    """
    Inputs: data: pandas dataframe of text for embedding with labels
            text: string, column name of text to be embedded
            labels: string, column name of text labels

    Extracts Word2Vec embeddings of text in a pandas dataframe, returns array
    of embedded vectors replacing text column
    """
    # Extract and return Word2Vec embeddings
    return Word2Vec.get_embeddings(data, text)


def get_elmo_embeddings(data, text):
    """
    Inputs: data: pandas dataframe of text for embedding with labels
            text: string, column name of text to be embedded
            labels: string, column name of text labels

    Extracts ELMo embeddings of text in a pandas dataframe, returns array of
    embedded vectors replacing text column
    """
    # Extract and return ELMo embeddings
    return ELMo.get_embeddings(data, text)


def get_fasttext_embeddings(data, text, labels):
    """
    Inputs: data: pandas dataframe of text for embedding with labels
            text: string, column name of text to be embedded
            labels: string, column name of text labels

    Extracts Fasttext embeddings of text in a pandas dataframe,
    returns array of embedded vectors replacing text column
    """
    # Extract and return fasttext embeddings
    return fasttext.fasttext_embedding(data, text, labels)


def get_glove_embeddings(data, text, labels):
    """
    Inputs: data: pandas dataframe of text for embedding with labels
            text: string, column name of text to be embedded
            labels: string, column name of text labels

    Extracts GLoVE embeddings of text in a pandas dataframe, returns arrray of
    embedded vectors replacing text column
    """
    # Extract and return GLoVE embeddings
    return glove.glove_embedding(data, text, labels)


def train_test_split_downstream(features, labels, test_size, random_state):
    """
    Inputs: features: numpy array of embedded text
            labels: string, column name of text labels
            test_size: float, fraction of data reserved for test split
            random_state, integer, the random state of the split

    Separates targets and labels into training and testing data using
    sci-kit learn train test split. Return four numpy arrays of split data
    """

    # Use train test split function on data, return data as np arrays
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def random_forest_model(X_train, X_test, y_train, y_test, n_estimators=1100, scoring_metric='accuracy'):
    """
    Inputs: X_train, numpy array, numpy array of training features
            X_test, numpy array, numpy array of testing features
            y_train, numpy array, numpy array of training labels
            y_test, numpy array, numpy array of testing labels,
            n_estimators, integer, number of trees in random forest
            scoring_metric, string, allows user to choose between sklearn
            scoring metrics "accuracy", "precision", "recall", or "auc"

    Implements sklearn random forest classifier on training and
    testing data, returns accuracy metric as float specified by user
    """

    # Build Random Forest Classifier Object
    rf = RandomForestClassifier(n_estimators=n_estimators).fit(X_train, y_train)

    # Make Predictions
    predictions = rf.predict(X_test)

    # If else statements that allow the user to specify their scoring metric
    if scoring_metric == 'accuracy':
        score = rf.score(X_test, y_test)
        return score
    elif scoring_metric == 'precision':
        precision = precision_score(y_test, predictions)
        return precision
    elif scoring_metric == 'recall':
        recall = recall_score(y_test, predictions)
        return recall
    elif scoring_metric == 'auc':
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        auc_score = auc(fpr, tpr)
        return auc_score


def decision_tree_model(X_train, X_test, y_train, y_test, scoring_metric='accuracy'):
    """
    Inputs: X_train, numpy array, numpy array of training features
            X_test, numpy array, numpy array of testing features
            y_train, numpy array, numpy array of training labels
            y_test, numpy array, numpy array of testing labels,
            scoring_metric, string, allows user to choose between sklearn
            scoring metrics "accuracy", "precision", "recall", or "auc"

    Implements sklearn decision tree classifier on training and
    testing data, returns accuracy metric as float specified by user
    """

    # Build Decision Tree Classifier Object
    clf_decision_tree = DecisionTreeClassifier()
    clf_decision_tree.fit(X_train, y_train)

    # Make Predictions
    predictions = clf_decision_tree.predict(X_test)

    # If else statements that allow the user to specify their scoring metric
    if scoring_metric == 'accuracy':
        score = accuracy_score(y_test, predictions)
        return score
    elif scoring_metric == 'precision':
        precision = precision_score(y_test, predictions)
        return precision
    elif scoring_metric == 'recall':
        recall = recall_score(y_test, predictions)
        return recall
    elif scoring_metric == 'auc':
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        auc_score = auc(fpr, tpr)
        return auc_score


def logistic_regression_model(X_train, X_test, y_train, y_test, scoring_metric='accuracy'):
    """
    Inputs: X_train, numpy array, numpy array of training features
            X_test, numpy array, numpy array of testing features
            y_train, numpy array, numpy array of training labels
            y_test, numpy array, numpy array of testing labels,
            scoring_metric, string, allows user to choose between sklearn
            scoring metrics "accuracy", "precision", "recall", or "auc"

    Implements sklearn logistic regression classifier on training
    and testing data, returns accuracy metric as float specified by user
    """

    # Build Logistic Classifer Object
    classifier = LogisticRegression(max_iter=100000)
    classifier.fit(X_train, y_train)

    # Make Predictions
    predictions = classifier.predict(X_test)

    # If else statements that allow the user to specify their scoring metric
    if scoring_metric == 'accuracy':
        score = accuracy_score(y_test, predictions)
        return score
    elif scoring_metric == 'precision':
        precision = precision_score(y_test, predictions)
        return precision
    elif scoring_metric == 'recall':
        recall = recall_score(y_test, predictions)
        return recall
    elif scoring_metric == 'auc':
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        auc_score = auc(fpr, tpr)
        return auc_score


def svm_model(X_train, X_test, y_train, y_test, scoring_metric='accuracy'):
    """
    Inputs: X_train, numpy array, numpy array of training features
            X_test, numpy array, numpy array of testing features
            y_train, numpy array, numpy array of training labels
            y_test, numpy array, numpy array of testing labels,
            scoring_metric, string, allows user to choose between sklearn
            scoring metrics "accuracy", "precision", "recall", or "auc"

    Implements sklearn support vector martrix classifier on
    training and testing data, returns accuracy metric as float specified
    by user
    """

    # Build SVM model
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # Make Predictions
    predictions = clf.predict(X_test)

    # If else statements that allow the user to specify their scoring metric
    if scoring_metric == 'accuracy':
        score = accuracy_score(y_test, predictions)
        return score
    elif scoring_metric == 'precision':
        precision = precision_score(y_test, predictions)
        return precision
    elif scoring_metric == 'recall':
        recall = recall_score(y_test, predictions)
        return recall
    elif scoring_metric == 'auc':
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        auc_score = auc(fpr, tpr)
        return auc_score
    

def find_optimal_method(data, features, labels, test_size=0.2, random_state=42, scoring_metric='accuracy'):
    """
    Inputs: data: pandas dataframe, pandas dataframe of text and labels for classification
            features: string, column name of text to be embedded
            labels: string, column name of text labels
            test_size: float, fraction of data reserved for test split
            random_state, integer, the random state of the split

    Runs all seven embedding methods against all four classifiers. Returns a pandas dataframe of each of the scores,
    allowing the user to determine the best classifier and embedding method combination for their data.

    Warning: with unintstalled pretrained embeddings this script can take well over an hour and a half to run
    """

    # Generate embeddings for all seven models
    bow_embeddings = get_BoW_embeddings(data, features, labels)
    tf_idf_embeddings = get_tfidf_embeddings(data, features,labels)
    bert_embeddings = get_bert_embeddings(data, features, labels)
    word2vec_embeddings = get_Word2Vec_embeddings(data, features)
    elmo_embeddings = get_elmo_embeddings(data, features)
    fasttext_embeddings = get_fasttext_embeddings(data, features, labels)
    glove_embeddings = get_glove_embeddings(data, features, labels)

    # Split the embeddings and labels for all seven embeddings
    X_train_bow, X_test_bow, y_train, y_test = train_test_split_downstream(bow_embeddings, data[labels], test_size=test_size, random_state=random_state)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split_downstream(tf_idf_embeddings, data[labels], test_size=test_size, random_state=random_state)
    X_train_bert, X_test_bert, y_train, y_test = train_test_split_downstream(bert_embeddings, data[labels], test_size=test_size, random_state=random_state)
    X_train_w2v, X_test_w2v, y_train, y_test = train_test_split_downstream(word2vec_embeddings, data[labels], test_size=test_size, random_state=random_state)
    X_train_elmo, X_test_elmo, y_train, y_test = train_test_split_downstream(elmo_embeddings, data[labels], test_size=test_size, random_state=random_state)
    X_train_fasttext, X_test_fasttext, y_train, y_test = train_test_split_downstream(fasttext_embeddings, data[labels], test_size=test_size, random_state=random_state)
    X_train_glove, X_test_glove, y_train, y_test = train_test_split_downstream(glove_embeddings, data[labels], test_size=test_size, random_state=random_state)

    # Iteratively go through each model, finding the accuracy for each classifier + model combination
    bow_embeddings_values = [
        random_forest_model(X_train_bow, X_test_bow, y_train, y_test, scoring_metric=scoring_metric),
        decision_tree_model(X_train_bow, X_test_bow, y_train, y_test, scoring_metric=scoring_metric),
        logistic_regression_model(X_train_bow, X_test_bow, y_train, y_test, scoring_metric=scoring_metric),
        svm_model(X_train_bow, X_test_bow, y_train, y_test, scoring_metric=scoring_metric)
    ]

    tf_idf_embeddings_values = [
        random_forest_model(X_train_tfidf, X_test_tfidf, y_train, y_test, scoring_metric=scoring_metric),
        decision_tree_model(X_train_tfidf, X_test_tfidf, y_train, y_test, scoring_metric=scoring_metric),
        logistic_regression_model(X_train_tfidf, X_test_tfidf, y_train, y_test, scoring_metric=scoring_metric),
        svm_model(X_train_tfidf, X_test_tfidf, y_train, y_test, scoring_metric=scoring_metric)
    ]

    bert_embeddings_values = [
        random_forest_model(X_train_bert, X_test_bert, y_train, y_test, scoring_metric=scoring_metric),
        decision_tree_model(X_train_bert, X_test_bert, y_train, y_test, scoring_metric=scoring_metric),
        logistic_regression_model(X_train_bert, X_test_bert, y_train, y_test, scoring_metric=scoring_metric),
        svm_model(X_train_bert, X_test_bert, y_train, y_test, scoring_metric=scoring_metric)
    ]

    w2v_embeddings_values = [
        random_forest_model(X_train_w2v, X_test_w2v, y_train, y_test, scoring_metric=scoring_metric),
        decision_tree_model(X_train_w2v, X_test_w2v, y_train, y_test, scoring_metric=scoring_metric),
        logistic_regression_model(X_train_w2v, X_test_w2v, y_train, y_test, scoring_metric=scoring_metric),
        svm_model(X_train_w2v, X_test_w2v, y_train, y_test, scoring_metric=scoring_metric)
    ]

    elmo_embeddings_values = [
        random_forest_model(X_train_elmo, X_test_elmo, y_train, y_test, scoring_metric=scoring_metric),
        decision_tree_model(X_train_elmo, X_test_elmo, y_train, y_test, scoring_metric=scoring_metric),
        logistic_regression_model(X_train_elmo, X_test_elmo, y_train, y_test, scoring_metric=scoring_metric),
        svm_model(X_train_elmo, X_test_elmo, y_train, y_test, scoring_metric=scoring_metric)
    ]

    fasttext_embeddings_values = [
        random_forest_model(X_train_fasttext, X_test_fasttext, y_train, y_test, scoring_metric=scoring_metric),
        decision_tree_model(X_train_fasttext, X_test_fasttext, y_train, y_test, scoring_metric=scoring_metric),
        logistic_regression_model(X_train_fasttext, X_test_fasttext, y_train, y_test, scoring_metric=scoring_metric),
        svm_model(X_train_fasttext, X_test_fasttext, y_train, y_test, scoring_metric=scoring_metric)
    ]

    glove_embeddings_values = [
        random_forest_model(X_train_glove, X_test_glove, y_train, y_test, scoring_metric=scoring_metric),
        decision_tree_model(X_train_glove, X_test_glove, y_train, y_test, scoring_metric=scoring_metric),
        logistic_regression_model(X_train_glove, X_test_glove, y_train, y_test, scoring_metric=scoring_metric),
        svm_model(X_train_glove, X_test_glove, y_train, y_test, scoring_metric=scoring_metric)
    ]

    # Return a pandas dataframe of each of the model's scores with each classifier
    score_pd = { "Classifier" : ['Random Forest',' Decision Tree','Logistic Regression', 'SVM'],
                'Bag of Words': bow_embeddings_values, 'tf idf': tf_idf_embeddings_values, 'BERT': bert_embeddings_values,
                'Word2Vec': w2v_embeddings_values, 'ELMo': elmo_embeddings_values, 'FastText': fasttext_embeddings_values,
                'GLoVE': glove_embeddings_values}

    return pd.DataFrame(data=score_pd)
