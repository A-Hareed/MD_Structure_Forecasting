import tensorflow as tf
from tensorflow.keras import layers

# Model parameters
img_height = 28  # Example image height
img_width = 28  # Example image width
channels = 1  # Example for grayscale images (change for RGB)
latent_dim = 32  # Dimension of latent space

# Encoder
encoder = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten()
])

# Latent layer
latent = layers.Dense(latent_dim)

# Decoder
decoder = tf.keras.Sequential([
    layers.Reshape((4, 4, 16)),  # Adjust based on latent_dim and desired output shape
    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(channels, (3, 3), activation='sigmoid', padding='same'),  # Adjust activation for your data type (sigmoid for 0-1 values)
])

# Define the forward pass (equivalent to call method in a class)
def autoencoder(x):
  encoded = encoder(x)
  latent = latent(encoded)
  decoded = decoder(latent)
  return decoded

# Compile the model (specify loss function and optimizer)
autoencoder = tf.keras.Model(inputs=tf.keras.Input(shape=(img_height, img_width, channels)), outputs=autoencoder(tf.keras.Input(shape=(img_height, img_width, channels))))
autoencoder.compile(loss='mse', optimizer='adam')




# Version 2
import tensorflow as tf
from tensorflow.keras import layers

# Model parameters
sequence_length = 100  # Example sequence length (adjust based on your data)
input_features = 10  # Example number of input features (adjust based on your data)
latent_dim = 32  # Dimension of latent space

# Encoder
encoder = tf.keras.Sequential([
  layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(sequence_length, input_features)),
  layers.MaxPooling1D(pool_size=2),
  layers.Flatten()
])

# Latent layer
latent = layers.Dense(latent_dim)

# Decoder (target output shape: (None, 108))
decoder = tf.keras.Sequential([
  layers.Dense(sequence_length * (108 // sequence_length)),  # Adjust based on desired final shape
  layers.Reshape((sequence_length, 108 // sequence_length)),
  layers.Conv1DTranspose(32, kernel_size=3, activation='relu', padding='same'),
  layers.UpSampling1D(pool_size=2),
  layers.Conv1D(input_features, kernel_size=3, activation='linear', padding='same'),  # Adjust activation for your data type
])

# Define the forward pass
def autoencoder(x):
  encoded = encoder(x)
  latent = latent(encoded)
  decoded = decoder(latent)
  return decoded

# Compile the model
autoencoder = tf.keras.Model(inputs=tf.keras.Input(shape=(sequence_length, input_features)), outputs=autoencoder(tf.keras.Input(shape=(sequence_length, input_features))))
autoencoder.compile(loss='mse', optimizer='adam')

