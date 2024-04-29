import numpy as np

# Assuming input_data and target_data are NumPy arrays
# input_data has shape (num_samples, num_input_features)
# target_data has shape (num_samples, num_target_features)

# Initialize an empty correlation matrix
num_input_features = input_data.shape[1]
num_target_features = target_data.shape[1]
correlation_matrix = np.zeros((num_input_features, num_target_features))

# Compute pairwise correlation
for i in range(num_input_features):
    for j in range(num_target_features):
        correlation_matrix[i, j] = np.corrcoef(input_data[:, i], target_data[:, j])[0, 1]

print("Pairwise Correlation Matrix:")
print(correlation_matrix)
