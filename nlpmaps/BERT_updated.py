import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.ensemble
import sklearn.model_selection
from transformers import BertTokenizer, BertModel
import torch
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset
def load_dataset():
    df1 = pd.read_excel('file_path', sheet_name='SamePerson Report')
    df2 = pd.read_excel('file_path', sheet_name='Multiple People Report')
    df3 = pd.read_excel('file_path', sheet_name='Multiple People Less Details')
    
    return df1, df2, df3

# Split dataset into train and test sets
def split_dataset(df):
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=206)
    
    return train, test

# Tokenize text using BERT tokenizer
def tokenize_text(text):
    input_ids = []
    attention_masks = []

    for sentence in text:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Train and test BERT model
def train_test_bert_model(train_text, train_label, test_text, test_label):
    train_text_list = train_text.tolist()
    train_text_str = [item for sublist in train_text_list for item in sublist]

    test_text_list = test_text.tolist()
    test_text_str = [item for sublist in test_text_list for item in sublist]

    train_label_list = train_label.tolist()
    train_label_str = [item for sublist in train_label_list for item in sublist]

    test_label_list = test_label.tolist()
    test_label_str = [item for sublist in test_label_list for item in sublist]

    # Tokenize the train dataset
    train_input_ids, train_attention_masks = tokenize_text(train_text_str)
    train_labels = torch.tensor(train_label_str)

    # Tokenize the test dataset
    test_input_ids, test_attention_masks = tokenize_text(test_text_str)
    test_labels = torch.tensor(test_label_str)

    # Load pre-trained BERT model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move data to the device
    train_input_ids = train_input_ids.to(device)
    train_attention_masks = train_attention_masks.to(device)
    train_labels = train_labels.to(device)

    test_input_ids = test_input_ids.to(device)
    test_attention_masks = test_attention_masks.to(device)
    test_labels = test_labels.to(device)

    # Forward pass through BERT model
    with torch.no_grad():
        train_outputs = model(train_input_ids, attention_mask=train_attention_masks)
        test_outputs = model(test_input_ids, attention_mask=test_attention_masks)

    train_features = train_outputs.pooler_output
    test_features = test_outputs.pooler_output

    # Convert features to numpy arrays
    train_x = train_features.cpu().numpy()
    test_x = test_features.cpu().numpy()
    train_y = train_labels.cpu().numpy()
    test_y = test_labels.cpu().numpy()

    return train_x, train_y, test_x, test_y

# Run grid search for a classifier
def run_grid_search(classifier_name, classifier, param_grid, train_x, train_y):
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(train_x, train_y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    return best_params, best_model

# Evaluate the model and plot the confusion matrix
def evaluate_model(model, test_x, test_y):
    pred_y = model.predict(test_x)
    accuracy = sklearn.metrics.accuracy_score(test_y, pred_y)
    confusion_mat = sklearn.metrics.confusion_matrix(test_y, pred_y)
    class_names = np.unique(test_y)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names)
    plt.show()
    
    return accuracy, confusion_mat

# Main function to run the entire process
def main():
    df1, df2, df3 = load_dataset()
    train_1, test_1 = split_dataset(df1)
    train_2, test_2 = split_dataset(df2)
    train_3, test_3 = split_dataset(df3)

    train_1_text = train_1['Report'].values.reshape(-1,1)
    test_1_text = test_1['Report'].values.reshape(-1,1)

    train_1_label = train_1['Level'].values.reshape(-1,1)
    test_1_label = test_1['Level'].values.reshape(-1,1)

    train_2_text = train_2['Report'].values.reshape(-1,1)
    test_2_text = test_2['Report'].values.reshape(-1,1)

    train_2_label = train_2['Level'].values.reshape(-1,1)
    test_2_label = test_2['Level'].values.reshape(-1,1)

    train_3_text = train_3['Report'].values.reshape(-1,1)
    test_3_text = test_3['Report'].values.reshape(-1,1)

    train_3_label = train_3['Level'].values.reshape(-1,1)
    test_3_label = test_3['Level'].values.reshape(-1,1)

    # Train and test BERT model for dataset 1
    train_x, train_y, test_x, test_y = train_test_bert_model(train_1_text, train_1_label, test_1_text, test_1_label)

    classifiers = {
        "Random Forest": {
            "model": sklearn.ensemble.RandomForestClassifier(),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 4, 6],
                "max_features": [0.5, 0.75, 1.0]
            }
        },
        "SVM": {
            "model": SVC(),
            "param_grid": {
                "C": [0.1, 1, 10],
                "kernel": ['linear', 'rbf'],
                "gamma": ['scale', 'auto']
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.1, 0.01, 0.001],
                "max_depth": [3, 5, 7]
            }
        },
        "Multi-Layer Perceptron": {
            "model": MLPClassifier(),
            "param_grid": {
                "hidden_layer_sizes": [(50,), (100,), (200,)],
                "activation": ['relu', 'tanh'],
                "solver": ['adam', 'sgd'],
                "learning_rate": ['constant', 'adaptive']
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.001]
            }
        },
        "Bagging": {
            "model": BaggingClassifier(),
            "param_grid": {
                "n_estimators": [10, 50, 100],
                "max_samples": [0.5, 0.75, 1.0],
                "max_features": [0.5, 0.75, 1.0]
            }
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 4, 6]
            }
        }
    }

    # Run grid search for each classifier
    results = {}

    for classifier_name, classifier_info in classifiers.items():
        print(f"Running grid search for {classifier_name}...")
        param_grid = classifier_info["param_grid"]
        classifier = classifier_info["model"]
        best_params, best_model = run_grid_search(classifier_name, classifier, param_grid, train_x, train_y)
        results[classifier_name] = {
            "best_params": best_params,
            "best_model": best_model
        }
        print(f"Best parameters for {classifier_name}: {best_params}")
        print(f"Best model for {classifier_name}: {best_model}")
        print()

        # Evaluate the best model
        accuracy, confusion_mat = evaluate_model(best_model, test_x, test_y)
        print(f"Accuracy for {classifier_name}: {accuracy}")
        print("Confusion Matrix:")
        print(confusion_mat)
        print("-------------------------------------------")
        print()

    # Plot confusion matrix for each classifier
    for classifier_name, result in results.items():
        best_model = result["best_model"]
        print(f"Confusion matrix for {classifier_name}:")
        _, _ = evaluate_model(best_model, test_x, test_y)
        print("-------------------------------------------")
        print()


# Run the main function
if __name__ == "__main__":
    main()
