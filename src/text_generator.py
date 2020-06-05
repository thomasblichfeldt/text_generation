import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import time


# Get path to file
from pathlib import Path
path = Path(__file__).parent / "..\\texts\\romeoandjuliet.txt"
save_path = Path(__file__).parent / "..\\models\\romeoandjuliet"

TRAINING = False
EPOCHS=50

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# ----- PREPROCESS DATA -----

raw_text = None
# Open file
with path.open() as f:
    raw_text = f.read() 

# The unique characters in the file
vocab = sorted(set(raw_text))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in raw_text])

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(raw_text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Create sequences for x and y data, 
# X being a sequance and y being that sequence + the one new letter that needs to be predicted.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Create X and y data
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Example of input and output
'''for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))'''

# Make training batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# ----- BUILD MODEL -----

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Build model structure

model = keras.Sequential()
if TRAINING:
    model.add(keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[BATCH_SIZE, None]))
else:
    model.add(keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]))
model.add(keras.layers.GRU(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
model.add(keras.layers.Dense(vocab_size))

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy())

# Name of the checkpoint files
checkpoint_prefix = os.path.join(save_path, "ckpt_{epoch}")

checkpoint_callback=keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

def generate_text(model, start_string, temperature=0.5):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

if TRAINING:
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
else:
    tf.train.latest_checkpoint(save_path)
    model.load_weights(tf.train.latest_checkpoint(save_path))
    model.build(tf.TensorShape([1, None]))
    model.summary()

    print(generate_text(model, start_string=u"ROMEO: ", temperature=0.2))
