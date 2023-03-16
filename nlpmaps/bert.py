import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

def download_and_extract_dataset(url, cache_dir='.'):
    """
    Downloads and extracts the dataset from a given URL using Keras' utility function `get_file`.
    
    Args:
        url (str): The URL to download the dataset from.
        cache_dir (str): The directory to cache the downloaded file (default is current directory).
        
    Returns:
        The path to the dataset directory.
        
    """
    # Download and extract the dataset using Keras' utility function `get_file`
    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                      untar=True, cache_dir=cache_dir,
                                      cache_subdir='')

    # Define the path to the dataset directory
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    # Define the path to the training data directory
    train_dir = os.path.join(dataset_dir, 'train')

    # Remove the 'unsup' directory that contains unsupervised data to make loading the data easier
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    
    return dataset_dir

def load_imdb_dataset(batch_size, validation_split, seed):
    """
    Loads the IMDB movie review dataset using Keras' text_dataset_from_directory function.
    
    Args:
        batch_size (int): The number of samples to use for each training batch.
        validation_split (float): The percentage of the data to use for validation.
        seed (int): The seed to use for shuffling the data.

    Returns:
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
        test_ds (tf.data.Dataset): The testing dataset.
        class_names (list): A list of the class names for the dataset.
    """
    # Define the Autotune constant to be used for data loading and preprocessing
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Load the training data from the directory 'aclImdb/train' using Keras' text_dataset_from_directory function
    # Set the batch size and validation split, as well as the subset to use for training
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=validation_split,
        subset='training',
        seed=seed)

    # Retrieve the class names from the training dataset
    class_names = raw_train_ds.class_names

    # Cache and prefetch the training dataset to improve performance during training
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Load the validation dataset from the directory 'aclImdb/train'
    # Set the batch size and validation split, as well as the subset to use for validation
    val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=validation_split,
        subset='validation',
        seed=seed)

    # Cache and prefetch the validation dataset to improve performance during validation
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Load the testing dataset from the directory 'aclImdb/test'
    # Set the batch size to be used during testing
    test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    # Cache and prefetch the testing dataset to improve performance during testing
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

def print_first_batch(train_ds, class_names):
    """
    Prints the first batch of the training dataset.

    Args:
        - train_ds: A TensorFlow dataset object representing the training dataset.
        - class_names: A list of class names representing the sentiment categories.

    Returns: 
        None.
    """

    # Loop over the first batch of the training dataset
    for text_batch, label_batch in train_ds.take(1):
        # Loop over the first 3 reviews in the batch
        for i in range(3):
            # Print the review text
            print(f'Review: {text_batch.numpy()[i]}')
            # Get the label for the review
            label = label_batch.numpy()[i]
            # Print the label and its corresponding class name
            print(f'Label: {label} ({class_names[label]})')

def select_bert_model(bert_model_name):
    """
    This function takes a string bert_model as input and returns a tuple of two callables: 
    a text encoder and a preprocessor function, both of which are TensorFlow Hub modules. 

    The text encoder is chosen based on the provided bert_model argument.
    The preprocessor is also chosen based on the bert_model and is used to preprocess the text data 
    for the encoder to consume.

    Args:
    - bert_model: A string representing the BERT model to be used. 
    
    Returns:
    - A tuple containing two callables: text_encoder and preprocessor.
    - text_encoder: A callable object that encodes text input using the specified BERT model.
    - preprocessor: A callable object that pre-processes text input in preparation for encoding by the specified BERT model.
    """
    map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
    }
    map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        }

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
    
    return tfhub_handle_encoder, tfhub_handle_preprocess

def preprocess_text(tfhub_handle_preprocess, text_test):
    """
    Preprocesses text using the preprocessing model provided by TensorFlow Hub for a given BERT model.

    Args:
        tfhub_handle_preprocess (str): The handle to the preprocessing model provided by TensorFlow Hub.
        text_test (list): A list of strings containing the text to be preprocessed.

    Returns:
        A dictionary containing the preprocessed text, including input_word_ids, input_mask, and input_type_ids.
    """
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    text_preprocessed = bert_preprocess_model(text_test)

    return text_preprocessed

def get_bert_outputs(text_preprocessed, tfhub_handle_encoder):
    """
    Loads a BERT model from TensorFlow Hub using the provided handle and wraps it in a KerasLayer.
    Passes preprocessed text through the BERT model to get outputs.
    
    Parameters:
    text_preprocessed (Tensor): Preprocessed text input for the BERT model.
    tfhub_handle_encoder (str): TensorFlow Hub handle for the BERT model to use.
    
    Returns:
    dict: A dictionary of the BERT outputs (pooled_output and sequence_output).
    """
    # Load the BERT model from TensorFlow Hub using the provided handle and wrap it in a KerasLayer
    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    
    # Pass preprocessed text through the BERT model to get outputs
    bert_results = bert_model(text_preprocessed)
    
    # Return a dictionary of the BERT outputs
    return {"pooled_output": bert_results["pooled_output"], 
            "sequence_output": bert_results["sequence_output"]}

def build_classifier_model(tfhub_handle_encoder, tfhub_handle_preprocess):
    """
    Builds and compiles a TensorFlow model for binary text classification using a pre-trained BERT model.

    Args:
        tfhub_handle_encoder (str): The handle to the BERT model provided by TensorFlow Hub.
        tfhub_handle_preprocess (str): The handle to the preprocessing model provided by TensorFlow Hub.

    Returns:
        A compiled TensorFlow model for binary text classification using BERT.
    """
    
    # Define input layer for the text
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    # Load the preprocessing layer from TF Hub
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    # Preprocess the text using the preprocessing layer
    encoder_inputs = preprocessing_layer(text_input)
    # Load the BERT model from TF Hub
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    # Pass the preprocessed inputs through the BERT model
    outputs = encoder(encoder_inputs)
    # Get the pooled output from BERT
    net = outputs['pooled_output']
    # Add a dropout layer
    net = tf.keras.layers.Dropout(0.1)(net)
    # Add a dense layer with sigmoid activation for binary classification
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    # Define the model input and output
    model = tf.keras.Model(text_input, net)
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

def build_loss_and_metrics():
    """
    Builds the BinaryCrossentropy loss and BinaryAccuracy metric for binary classification.

    Returns:
        A tuple containing the BinaryCrossentropy loss and BinaryAccuracy metric.
    """
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    return loss, metrics

def create_optimizer(init_lr, num_train_steps, num_warmup_steps, optimizer_type='adamw'):
    """
    Create and return an optimizer with the specified parameters.

    Args:
        init_lr (float): The initial learning rate for the optimizer.
        num_train_steps (int): The total number of training steps.
        num_warmup_steps (int): The number of warmup steps.
        optimizer_type (str): The type of optimizer to use (default is 'adamw').

    Returns:
        The created optimizer.
    """
    return optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type=optimizer_type)


def plot_history(history):
    """
    Plots the training and validation loss and accuracy for a given training history.

    Args:
        history (tf.keras.callbacks.History): The training history to plot.

    Returns:
        None.
    """
    history_dict = history.history

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

    # Save the plots as JPG
    plt.savefig('training_and_validation_loss.jpg')
    plt.savefig('training_and_validation_accuracy.jpg')
    
def save_model(model, dataset_name, saved_model_path):
    """
    Saves a trained model to a given file path.

    Args:
        model: A TensorFlow model to be saved.
        dataset_name (str): The name of the dataset used to train the model.
        saved_model_path (str): The path to save the model file.

    Returns:
        None.
    """
    saved_model_path = saved_model_path.format(dataset_name.replace('/', '_'))
    model.save(saved_model_path, include_optimizer=False)

def print_my_examples(inputs, results):
    """
    Prints the inputs along with their corresponding prediction scores.

    Args:
    - inputs: A list of input strings.
    - results: A list of prediction scores.

    Returns: None.
    """
    result_for_printing = [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}' for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()

