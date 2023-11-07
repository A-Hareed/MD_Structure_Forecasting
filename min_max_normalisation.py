# Imports 
import os,sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# load cartesian coordinate file. 
file_name = 'extracted_pdb_trajectory_more_config_recentering_aligned.npy'
full_data = np.load(filename)

# Select the size of the training set and validation set

if int(sys.argv[1]) >0:
  training_size = int(sys.argv[1])

val_size = int(input('Insert Validation dataset Size(by number of frames): '))


training_data = full_data[:training_size,1:]
val_data = full_data[training_size:val_size,1:]


num_samples, num_features = training_data.shape
num_samples_val, num_features_val = val_data.shape




data_2d = training_data.reshape(num_samples * num_features, 1)

data_2d_val = val_data.reshape(num_samples_val * num_features_val, 1)

scaler = MinMaxScaler()
scaler.fit(data_2d)

data_2d = scaler.transform(data_2d)
data_2d_val = scaler.transform(data_2d_val)

training_data = data_2d.reshape(num_samples, num_features)

val_data = data_2d_val.reshape(num_samples_val, num_features_val)




print(training_data.shape,val_data.shape)

training_filename = f'training_dataset_size_{training_size}.npy'
validation_filename = f'validation_dataset_size_{val_size}.npy'

np.save(training_filename, training_data)
np.save(validation_filename, val_data)
