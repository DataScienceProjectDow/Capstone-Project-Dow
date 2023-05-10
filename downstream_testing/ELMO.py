import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

main_data = pd.read_csv('IMDBDataset.csv.zip')
main_data['sentiment'] = main_data['sentiment'].map({'positive':1, 'negative': 0})
elmo = hub.load('/Users/andrewsimon/Downloads/elmo_3').signatures['default']
X_train, X_test, y_train, y_test = train_test_split(main_data['review'], main_data['sentiment'], test_size=0.2, random_state=1516)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
input_tensor_train = X_train
input_tensor_test = X_test
embeddings_tensor_train = elmo(tf.constant(input_tensor_train))['elmo']
embeddings_tensor_test = elmo(tf.constant(input_tensor_test))['elmo']
embeddings_train = embeddings_tensor_train.numpy()
embeddings_test = embeddings_tensor_test.numpy()
training_padded = pad_sequences(embeddings_train, maxlen=120, truncating='post')
testing_padded = pad_sequences(embeddings_tensor_test, maxlen=120, truncating='post')

batch = 32

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', 'Recall', 'AUC', 'Precision', 'FalseNegatives', 'FalsePositives'])
    
num_epochs = 10
model.fit(training_padded, y_train, epochs=num_epochs,batch_size=batch, validation_data=(testing_padded, y_test))
model.summary()