import tensorflow as tf

# Define your deeper model architecture
def create_deeper_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),  # Adding another hidden layer with 1024 neurons
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='linear')  # Linear activation for regression
    ])
    return model

# Create the deeper model
input_shape = (3,)  # Assuming you have 3 input features
output_shape = 12   # Number of target variables
deeper_model = create_deeper_model(input_shape, output_shape)

# Compile the model
deeper_model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # You can change the loss and metrics as needed

# Train the deeper model with your data
# deeper_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))



# SVM
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

# Create an SVR model
svr = SVR(kernel='rbf')  # You can choose different kernels based on your data

# Wrap the SVR model with MultiOutputRegressor for multi-output regression
svm_model = MultiOutputRegressor(svr)

# Train the model with your data
# svm_model.fit(X_train, y_train)
