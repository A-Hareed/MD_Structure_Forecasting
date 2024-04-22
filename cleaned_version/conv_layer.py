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
