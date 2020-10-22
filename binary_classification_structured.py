# Import TensorFlow and other libraries
# pip install -q sklearn

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Use Pandas to create a dataframe
# Pandas is a Python library with many helpful utilities for loading and working with structured data.
# You will use Pandas to download the dataset from a URL, and load it into a dataframe.

import pathlib

dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                        extract=True, cache_dir='.')
dataframe = pd.read_csv(csv_file)

# Create target variable
# After modifying the label column, 0 will indicate the pet was not adopted, and 1 will indicate it was.
#
# # In the original dataset "4" indicates the pet was not adopted.
dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)
#
# # Drop un-used columns.
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

# Split the dataframe into train, validation, and test

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#7383 train examples
#1846 validation examples
#2308 test examples

# Create an input pipeline using tf.data
# We wrap the dataframes with tf.data, in order to shuffle and batch the data.

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch )

#Every feature: ['Type', 'Age', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt']
#A batch of ages: tf.Tensor([ 2 36  1 30  2], shape=(5,), dtype=int64)
#A batch of targets: tf.Tensor([0 1 1 1 0], shape=(5,), dtype=int64)

# Numeric columns

# For each of the Numeric feature, you will use a Normalization() layer to make sure the mean of each feature is 0 and its standard deviation is 1.
# get_normalization_layer function returns a layer which applies featurewise normalization to numerical features.

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for our feature.
  normalizer = preprocessing.Normalization()

  # Prepare a Dataset that only yields our feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

photo_count_col = train_features['PhotoAmt']
layer = get_normalization_layer('PhotoAmt', train_ds)
layer(photo_count_col)

#<tf.Tensor: shape=(5, 1), dtype=float32, numpy=
#array([[ 3.9776573 ],
#       [ 0.44618952],
#       [ 1.0882746 ],
#       [ 0.12514699],
#       [-0.83798057]], dtype=float32)>

# Categorical columns

#In this dataset, Type is represented as a string (e.g. 'Dog', or 'Cat').
# You cannot feed strings directly to a model.
# The preprocessing layer takes care of representing strings as a one-hot vector.
# get_category_encoding_layer function returns a layer which maps values from a vocabulary to integer indices and one-hot encodes the features.

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a StringLookup layer which will turn strings into integer indices
  if dtype == 'string':
    index = preprocessing.StringLookup(max_tokens=max_tokens)
  else:
    index = preprocessing.IntegerLookup(max_values=max_tokens)

  # Prepare a Dataset that only yields our feature
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Create a Discretization for our integer indices.
  encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

  # Prepare a Dataset that only yields our feature.
  feature_ds = feature_ds.map(index)

  # Learn the space of possible indices.
  encoder.adapt(feature_ds)

  # Apply one-hot encoding to our indices. The lambda function captures the
  # layer so we can use them, or include them in the functional model later.
  return lambda feature: encoder(index(feature))

type_col = train_features['Type']
layer = get_category_encoding_layer('Type', train_ds, 'string')
layer(type_col)

#<tf.Tensor: shape=(5, 4), dtype=float32, numpy=
#array([[0., 0., 0., 1.],
#       [0., 0., 1., 0.],
#       [0., 0., 0., 1.],
#       [0., 0., 0., 1.],
#       [0., 0., 1., 0.]], dtype=float32)>

#Often, you don't want to feed a number directly into the model, but instead use a one-hot encoding of those inputs.
# Consider raw data that represents a pet's age.
type_col = train_features['Age']
category_encoding_layer = get_category_encoding_layer('Age', train_ds,
                                                      'int64', 5)
category_encoding_layer(type_col)

#<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
#array([[0., 0., 1., 0., 0.],
#       [0., 1., 0., 0., 0.],
#       [0., 0., 0., 0., 1.],
#       [0., 1., 0., 0., 0.],
#       [0., 0., 1., 0., 0.]], dtype=float32)>

# Choose which columns to use

#Earlier, we used a small batch size to demonstrate the input pipeline. Let's now create a new input pipeline with a larger batch size.

batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

all_inputs = []
encoded_features = []

# Numeric features.
for header in ['PhotoAmt', 'Fee']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

# Categorical features encoded as integers.
age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')
encoding_layer = get_category_encoding_layer('Age', train_ds, dtype='int64',
                                             max_tokens=5)
encoded_age_col = encoding_layer(age_col)
all_inputs.append(age_col)
encoded_features.append(encoded_age_col)

# Categorical features encoded as string.
categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)

#Create, compile, and train the model
#Now we can create our end-to-end model.

all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Let's visualize the connectivity graph:

# rankdir='LR' is used to make the graph horizontal.

tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

#Train the model.

model.fit(train_ds, epochs=10, validation_data=val_ds)

#Epoch 1/10
#29/29 [==============================] - 1s 20ms/step - loss: 0.6747 - accuracy: 0.5123 - val_loss: 0.5660 - val_accuracy: 0.6983
#Epoch 2/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.6025 - accuracy: 0.6469 - val_loss: 0.5478 - val_accuracy: 0.7275
#Epoch 3/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5825 - accuracy: 0.6584 - val_loss: 0.5355 - val_accuracy: 0.7373
#Epoch 4/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5658 - accuracy: 0.6844 - val_loss: 0.5259 - val_accuracy: 0.7432
#Epoch 5/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5605 - accuracy: 0.6841 - val_loss: 0.5197 - val_accuracy: 0.7481
#Epoch 6/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5564 - accuracy: 0.6912 - val_loss: 0.5163 - val_accuracy: 0.7524
#Epoch 7/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5465 - accuracy: 0.7008 - val_loss: 0.5131 - val_accuracy: 0.7470
#Epoch 8/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5443 - accuracy: 0.7046 - val_loss: 0.5104 - val_accuracy: 0.7508
#Epoch 9/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5350 - accuracy: 0.7198 - val_loss: 0.5082 - val_accuracy: 0.7524
#Epoch 10/10
#29/29 [==============================] - 0s 6ms/step - loss: 0.5358 - accuracy: 0.7139 - val_loss: 0.5067 - val_accuracy: 0.7546

#<tensorflow.python.keras.callbacks.History at 0x7fef499709b0>

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

#10/10 [==============================] - 0s 4ms/step - loss: 0.5143 - accuracy: 0.7439
#Accuracy 0.743934154510498

# Inference on new data
# You can now save and reload the Keras model.

model.save('my_pet_classifier')
reloaded_model = tf.keras.models.load_model('my_pet_classifier')

# To get a prediction for a new sample, you can simply call model.predict().
# There are just two things you need to do:

# 1. Wrap scalars into a list so as to have a batch dimension (models only process batches of data, not single samples).
# 2. Call convert_to_tensor on each feature.

sample = {
    'Type': 'Cat',
    'Age': 3,
    'Breed1': 'Tabby',
    'Gender': 'Male',
    'Color1': 'Black',
    'Color2': 'White',
    'MaturitySize': 'Small',
    'FurLength': 'Short',
    'Vaccinated': 'No',
    'Sterilized': 'No',
    'Health': 'Healthy',
    'Fee': 100,
    'PhotoAmt': 2,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = reloaded_model.predict(input_dict)
prob = tf.nn.sigmoid(predictions[0])

print(
    "This particular pet had a %.1f percent probability "
    "of getting adopted." % (100 * prob)
)

# This particular pet had a 83.4 percent probability of getting adopted.