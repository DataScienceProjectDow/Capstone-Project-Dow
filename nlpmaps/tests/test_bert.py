import os.path
import shutil
import tempfile
import unittest
import io
import sys
import tensorflow as tf
import tensorflow_hub as hub
from unittest.mock import MagicMock, patch
from bert import *


class DownloadAndExtractDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_download_and_extract_dataset(self):
        # Download and extract the dataset
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        dataset_dir = download_and_extract_dataset(url, cache_dir=self.tmp_dir.name)

        # Check that the dataset directory exists
        self.assertTrue(os.path.isdir(dataset_dir))

        # Check that the training directory exists
        train_dir = os.path.join(dataset_dir, 'train')
        self.assertTrue(os.path.isdir(train_dir))

        # Check that the 'unsup' directory was removed
        unsup_dir = os.path.join(train_dir, 'unsup')
        self.assertFalse(os.path.exists(unsup_dir))

        # Check that the 'aclImdb_v1.tar.gz' file was downloaded and extracted
        tar_file = os.path.join(self.tmp_dir.name, '.keras', 'datasets', 'aclImdb_v1.tar.gz')
        self.assertTrue(os.path.isfile(tar_file))

        # Check that the files in the dataset directory are the expected ones
        expected_files = ['README', 'imdb.vocab', 'imdbEr.txt', 'test', 'train']
        actual_files = os.listdir(dataset_dir)
        self.assertSetEqual(set(actual_files), set(expected_files))

        # Check that the files in the training directory are the expected ones
        expected_files = ['pos', 'neg', 'unsupBow.feat', 'unsup', 'urls_pos.txt', 'urls_neg.txt']
        actual_files = os.listdir(train_dir)
        self.assertSetEqual(set(actual_files), set(expected_files))

        # Check that the 'pos' directory contains files
        pos_dir = os.path.join(train_dir, 'pos')
        self.assertGreater(len(os.listdir(pos_dir)), 0)

        # Check that the 'neg' directory contains files
        neg_dir = os.path.join(train_dir, 'neg')
        self.assertGreater(len(os.listdir(neg_dir)), 0)
        
class TestLoadIMDBDataset(unittest.TestCase):
    def test_batch_size(self):
        # Check that the batch size of the returned datasets matches the expected value
        batch_size = 32
        _, _, test_ds, _ = load_imdb_dataset(batch_size, 0.2, 123)
        for x, y in test_ds:
            self.assertEqual(x.shape[0], batch_size)
            break
    
    def test_validation_split(self):
        # Check that the validation split of the returned datasets matches the expected value
        validation_split = 0.2
        train_ds, val_ds, _, _ = load_imdb_dataset(32, validation_split, 123)
        train_size = train_ds.cardinality().numpy()
        val_size = val_ds.cardinality().numpy()
        total_size = train_size + val_size
        expected_val_size = int(total_size * validation_split)
        self.assertEqual(val_size, expected_val_size)
    
    def test_class_names(self):
        # Check that the class names of the returned datasets are correct
        _, _, test_ds, expected_class_names = load_imdb_dataset(32, 0.2, 123)
        class_names = test_ds.class_names
        self.assertListEqual(class_names, expected_class_names)
    
    def test_prefetching(self):
        # Check that the returned datasets are cached and prefetched for improved performance
        train_ds, val_ds, test_ds, _ = load_imdb_dataset(32, 0.2, 123)
        self.assertTrue(train_ds._is_cached)
        self.assertTrue(train_ds._prefetcher._buffer_size > 0)
        self.assertTrue(val_ds._is_cached)
        self.assertTrue(val_ds._prefetcher._buffer_size > 0)
        self.assertTrue(test_ds._is_cached)
        self.assertTrue(test_ds._prefetcher._buffer_size > 0)
        
class TestPrintFirstBatch(unittest.TestCase):
    def setUp(self):
        # Set up a small test dataset with 3 reviews and 2 classes
        self.train_ds = tf.data.Dataset.from_tensor_slices((
            ['Review 1', 'Review 2', 'Review 3'],
            [0, 1, 0]
        ))
        self.class_names = ['Negative', 'Positive']

    def test_print_first_batch(self):
        # Redirect stdout to a buffer
        with io.StringIO() as buf, redirect_stdout(buf):
            # Call the function with the test dataset and class names
            print_first_batch(self.train_ds, self.class_names)
            # Get the output from stdout
            output = buf.getvalue().strip()
            # Check that the output matches the expected format
            expected_output = 'Review: Review 1\nLabel: 0 (Negative)\nReview: Review 2\nLabel: 1 (Positive)\nReview: Review 3\nLabel: 0 (Negative)'
            self.assertEqual(output, expected_output)
class TestPreprocessText(unittest.TestCase):

    @patch('my_module.hub.KerasLayer')
    def test_preprocess_text(self, mock_keras_layer):
        # Setup mock object
        mock_bert_preprocess_model = MagicMock()
        mock_keras_layer.return_value = mock_bert_preprocess_model
        mock_text_test = ["Hello, world!", "How are you?"]

        # Test preprocessing
        result = preprocess_text("test_handle", mock_text_test)

        # Assert expected results
        self.assertEqual(result.keys(), ["input_word_ids", "input_mask", "input_type_ids"])
        mock_bert_preprocess_model.assert_called_once_with(mock_text_test)

class TestGetBertOutputs(unittest.TestCase):
    
    def setUp(self):
        self.text = ["This is a test.", "This is another test."]
        self.tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
        
    def test_outputs_dict_keys(self):
        text_preprocessed = {"input_word_ids": tf.constant([[101, 2023, 2003, 1037, 3231, 1012, 102], 
                                                            [101, 2023, 2003, 2178, 3231, 1012, 102]]),
                             "input_mask": tf.constant([[1, 1, 1, 1, 1, 1, 1], 
                                                        [1, 1, 1, 1, 1, 1, 1]]),
                             "input_type_ids": tf.constant([[0, 0, 0, 0, 0, 0, 0], 
                                                            [0, 0, 0, 0, 0, 0, 0]])}
        bert_outputs = get_bert_outputs(text_preprocessed, self.tfhub_handle_encoder)
        self.assertTrue("pooled_output" in bert_outputs.keys())
        self.assertTrue("sequence_output" in bert_outputs.keys())
        
    def test_outputs_shape(self):
        text_preprocessed = {"input_word_ids": tf.constant([[101, 2023, 2003, 1037, 3231, 1012, 102], 
                                                            [101, 2023, 2003, 2178, 3231, 1012, 102]]),
                             "input_mask": tf.constant([[1, 1, 1, 1, 1, 1, 1], 
                                                        [1, 1, 1, 1, 1, 1, 1]]),
                             "input_type_ids": tf.constant([[0, 0, 0, 0, 0, 0, 0], 
                                                            [0, 0, 0, 0, 0, 0, 0]])}
        bert_outputs = get_bert_outputs(text_preprocessed, self.tfhub_handle_encoder)
        self.assertEqual(bert_outputs["pooled_output"].shape, (2, 768))
        self.assertEqual(bert_outputs["sequence_output"].shape, (2, 7, 768))

class TestClassifierModel(unittest.TestCase):

    def setUp(self):
        self.tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
        self.tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        self.model = build_classifier_model(self.tfhub_handle_encoder, self.tfhub_handle_preprocess)

    def test_model_output_shape(self):
        input_shape = (1,)
        input_text = ["This is a test sentence."]
        expected_output_shape = (1, 1)
        self.assertEqual(self.model(tf.constant(input_text)).shape, expected_output_shape)

    def test_model_loss_function(self):
        expected_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.assertEqual(self.model.loss, expected_loss)

    def test_model_metrics(self):
        expected_metrics = [tf.keras.metrics.BinaryAccuracy()]
        self.assertEqual(self.model.metrics, expected_metrics)

class TestBuildLossAndMetrics(unittest.TestCase):

    def test_build_loss(self):
        loss, _ = build_loss_and_metrics()
        self.assertIsInstance(loss, tf.keras.losses.BinaryCrossentropy)

    def test_build_metrics(self):
        _, metrics = build_loss_and_metrics()
        self.assertIsInstance(metrics, tf.keras.metrics.BinaryAccuracy)

class TestCreateOptimizer(unittest.TestCase):
    
    def test_optimizer_type_default(self):
        # Test that the default optimizer type is 'adamw'
        optimizer = create_optimizer(init_lr=0.001, num_train_steps=100, num_warmup_steps=10)
        self.assertEqual(optimizer.get_config()['name'], 'AdamWeightDecay')
        
    def test_optimizer_type_sgd(self):
        # Test that the optimizer type can be set to 'sgd'
        optimizer = create_optimizer(init_lr=0.01, num_train_steps=200, num_warmup_steps=20, optimizer_type='sgd')
        self.assertEqual(optimizer.get_config()['name'], 'SGD')
        
    def test_optimizer_lr(self):
        # Test that the optimizer has the specified initial learning rate
        optimizer = create_optimizer(init_lr=0.002, num_train_steps=150, num_warmup_steps=15)
        self.assertAlmostEqual(optimizer.get_config()['learning_rate'], 0.002)
        
    def test_optimizer_steps(self):
        # Test that the optimizer has the correct number of total and warmup steps
        optimizer = create_optimizer(init_lr=0.001, num_train_steps=120, num_warmup_steps=12)
        self.assertEqual(optimizer.get_config()['total_steps'], 120)
        self.assertEqual(optimizer.get_config()['warmup_steps'], 12)

class TestPlotHistory(unittest.TestCase):

    def setUpClass(cls):
        # create a fake training history for testing
        cls.history = tf.keras.callbacks.History()
        cls.history.history = {
            'loss': [0.5, 0.3, 0.2],
            'val_loss': [0.4, 0.3, 0.2],
            'binary_accuracy': [0.8, 0.9, 0.95],
            'val_binary_accuracy': [0.75, 0.85, 0.9]
        }

    def test_plot_history_calls_matplotlib_subplots(self):
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            plot_history(self.history)
            mock_subplots.assert_called_once_with(nrows=2, ncols=1, figsize=(10, 6), tight_layout=True)

    def test_plot_history_calls_matplotlib_show(self):
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_history(self.history)
            mock_show.assert_called_once()

    def test_plot_history_calls_matplotlib_savefig(self):
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_history(self.history)
            mock_savefig.assert_has_calls([call('training_and_validation_loss.jpg'), call('training_and_validation_accuracy.jpg')])        

class TestSaveModel(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.saved_model_path = os.path.join(self.temp_dir, "{}.h5")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_model(self):
        # Create a simple model
        inputs = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(8, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Save the model
        dataset_name = "test_dataset"
        save_model(model, dataset_name, self.saved_model_path)
        
        # Check that the saved model file exists
        expected_path = os.path.join(self.temp_dir, "test_dataset.h5")
        self.assertTrue(os.path.exists(expected_path))
    
    def test_save_model_with_slash(self):
        # Create a simple model
        inputs = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(8, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Save the model with a dataset name that contains a slash
        dataset_name = "test/dataset"
        save_model(model, dataset_name, self.saved_model_path)
        
        # Check that the saved model file exists
        expected_path = os.path.join(self.temp_dir, "test_dataset.h5")
        self.assertTrue(os.path.exists(expected_path))
        


class TestPrintMyExamples(unittest.TestCase):
    def test_print_my_examples(self):
        inputs = ['This is a positive example', 'This is a negative example']
        results = [[0.9], [0.1]]

        expected_output = 'input: This is a positive example     : score: 0.900000\ninput: This is a negative example     : score: 0.100000\n\n'

        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            print_my_examples(inputs, results)
            self.assertEqual(fake_stdout.getvalue(), expected_output)

