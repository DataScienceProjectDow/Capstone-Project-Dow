# Import Libraries
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def to_pd_dataframe(file):
    """Converts a csv file to pandas dataframe. Expects a string path to
    csv and returns a pandas dataframe"""

    data = pd.read_csv(file)

    return data


def pre_process(data_processed, feature, sentiment, outcome_1, outcome_2):
    """Preprocesses the data by using word2vec preprocessing function.
    Used only for binary sentiment classification. Expects a pandas dataframe
    to be processed, the column name of the feature, the column name of the
    sentiment, the name (str) of outcome 1, and the name of outcome 2.
    Returns a processed data frame """

    # Add a "cleaned" column to the dataset which has been tokenized,
    # removed stop words, and had HTML data cut using Gensim's built
    # in preprocessor
    data_processed[feature + '_clean'] = data_processed[feature].apply(
        lambda x: gensim.utils.simple_preprocess(x))

    # Mapping binary sentiment to 1's and 0's
    data_processed[sentiment] = data_processed[sentiment].map(
        {outcome_1: 1, outcome_2: 0})

    return data_processed


def sample_data(data, num_samples):
    """Samples data from processed dataframe. Expects a pandas dataframe
    and number of samples (int), returns sampled dataframe"""

    # Taking the sample of the data
    sampled_data = data.sample(n=num_samples)

    return sampled_data


def data_split(data, feature, sentiment, testing_size=0.2):
    """Splits data into training and testing subsets. Expects a pandas
    dataframe to be split, the column name of the feature, the column
    name of the sentiment, and optional parameter (float) for the
    testing ratio. Returns X_train, X_test, y_train, and y_test arrays"""

    # Scikit learn train test split function
    X_train, X_test, y_train, y_test = train_test_split(
                                                        data[feature
                                                             + '_clean'],
                                                        data[sentiment],
                                                        test_size=testing_size
                                                        )

    return X_train, X_test, y_train, y_test


def vectorize(X_training, X_testing, vec_size=100, win=5, minimum_count=2):
    """Vectorizes the training and testing data using word2vec model.
    Expects X_training and X_testing data (arrays) along with default w2v
    parameters. Please consult the gensim handbook for more information about
    w2v parameters. Returns vectorized X arrays for use with classification
    models """

    # Building Word2Vec model
    w2v_model = gensim.models.Word2Vec(X_training,
                                       vector_size=vec_size,
                                       window=win,
                                       min_count=minimum_count)

    # Getting the list of words in our model with no duplicates
    words = list(set(w2v_model.wv.index_to_key))

    # Creating an empty dictionary for our vectors
    word_vec_dict = {}

    # Loop over dictionary and add vector for each word
    for word in words:
        word_vec_dict[word] = w2v_model.wv[word]

    word_df = pd.DataFrame(data=word_vec_dict)

    X_train_vect = []

    sample_counter = 0

    print('Vectorizing Data')

    # Our data is in what is known as a ragged array, meaning the array
    # is not N x N shaped. To resolve these differences the average
    # value for each vector position is taken and then appended to a
    # vectorized array in order to standardize the data

    # Loop over reviews in training data
    for review in X_training:
        # This code can take 10+ minutes to run for even a few
        # thousand samples this counter is to give the user
        # some sort of indication on far along the loop is
        sample_counter = sample_counter + 1
        print(str(sample_counter) + '/' + str(len(X_training) +
                                              len(X_testing)) +
                                    ' samples vectorized')
        # This counter is for taking the mean value of the column
        count = 0
        # Blank array to make the matrix math work out
        word_vec_avg = np.zeros(100)
        for word in review:
            if word in words:
                # If the word isn't in our corpus, we just throw it out
                count = count + 1
                # Sum of arrays
                word_vec_avg = word_vec_avg + word_df[word]
        # Taking the average
        word_vec_avg = word_vec_avg / count
        X_train_vect.append(word_vec_avg)

    X_test_vect = []

    # Repeating the exact step above but for testing data
    for review in X_testing:
        sample_counter = sample_counter + 1
        print(str(sample_counter) + '/' +
              str(len(X_training) + len(X_testing)) + ' samples vectorized')
        count = 0
        word_vec_avg = np.zeros(100)
        for word in review:
            if word in words:
                count = count + 1
                word_vec_avg = word_vec_avg + word_df[word]
        word_vec_avg = word_vec_avg / count
        X_test_vect.append(word_vec_avg)

    return X_train_vect, X_test_vect


def make_prediction(X_training_vect, X_testing_vect, y_train, y_test):
    """Uses sklearn classifier to fit, classify, and predict data.
    Expects vectorized (array) training and testing data, training
    and testing target data (array). Returns float accuracy score"""

    # Building a DTC model
    clf_decision_word2vec = DecisionTreeClassifier()

    # Fitting data
    clf_decision_word2vec.fit(X_training_vect, y_train)

    # Training data
    predictions = clf_decision_word2vec.predict(X_testing_vect)

    # Getting the accuracy of the model
    acc_score = accuracy_score(y_test, predictions)

    return acc_score


def accuracy_prediction(data, feature, sentiment, outcome_1, outcome_2,
                        number_samples, vec_size=100, win=5, minimum_count=2,
                        testing_size=0.2):
    """Wrapped function that returns the accuracy of the model
    trained on the dataset.Expects a pandas dataframe for the dataset,
    the feature (string) to be trained on, the target sentiment,
    the binary outcomes (string) of the sentiment, the number of samples (int)
    , along with default parameters for the word2vec model and testing
    size. Returns None, prints the accuracy score of the model.
    This function is meant to be used as a downstream diagnostic tool
    rather than standalone function."""

    # Wrapped Functions
    data_pd = to_pd_dataframe(data)
    data_pre_processed = pre_process(data_pd,
                                     feature, sentiment, outcome_1, outcome_2)
    sampled_data = sample_data(data_pre_processed, number_samples)
    X_train, X_test, y_train, y_test = data_split(sampled_data, feature,
                                                  sentiment, testing_size)
    X_train_vect, X_test_vect = vectorize(X_train, X_test, vec_size=vec_size,
                                          win=win, minimum_count=minimum_count
                                          )
    acc_score = make_prediction(X_train_vect, X_test_vect, y_train, y_test)

    # Printing the accuracy score
    print('accuracy score of Word2vec model ' + str(acc_score))
    return acc_score
