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

from sklearn.preprocessing import StandardScaler
# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(your_target_data)



#*****************************************************************
#*****************************************************************
#*****************************************************************
#*****************************************************************

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error




# SVM
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

# Create an SVR model
svr = SVR(kernel='rbf' ,verbose=True)  # You can choose different kernels based on your data

# Wrap the SVR model with MultiOutputRegressor for multi-output regression
svm_model = MultiOutputRegressor(svr)

# Train the model with your data
# svm_model.fit(X_train, y_train)



# Define your deeper model architecture
def create_deeper_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.LayerNormalization(),
        # tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(output_shape, activation='linear')  # Linear activation for regression
    ])
    return model

# Create the deeper model
input_shape = (3,)  # Assuming you have 3 input features
output_shape = 12   # Number of target variables
# deeper_model = create_deeper_model(input_shape, output_shape)


# Data Loading
data_bb = np.load('data_bb.npy')
data_aa = np.load('change_aa.npy')
print('tareget and feature shape',data_aa.shape,data_bb.shape)

train_feature_bb = data_bb[:30,:].reshape(-1,3)
val_feature_bb = data_bb[30:40,:].reshape(-1,3)
print('training and validation feature shape: ',train_feature_bb.shape, val_feature_bb.shape)




def get_labels(arr):
    start = -12
    for i in range(0,arr.shape[1],12):
        start = i
        end  = i+12
        print(start,end)
        if i ==0:
            label = arr[:,start:end]
        else:
            label = np.concatenate((label,arr[:,start:end]),axis=0)
    
    return label



# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_aa)



train_label = get_labels(scaled_data[:30,:])
val_label = get_labels(scaled_data[30:40,:])
print('train and valdiation shapes',train_label.shape, val_label.shape)


# Train the model with your data
svm_model.fit(train_feature_bb, train_label)





# Predict target values on validation set
y_pred = svm_model.predict(val_feature_bb)

# Calculate squared errors
squared_errors = np.square(y_pred - val_label)

# Average squared errors
mean_squared_error = np.mean(squared_errors)

# Root Mean Squared Error (RMSE)
validation_rmse = np.sqrt(mean_squared_error)

# Print the validation RMSE
print("Validation RMSE:", validation_rmse)


exit()
