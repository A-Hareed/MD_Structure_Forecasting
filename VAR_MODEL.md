
Vector Autoregression (VAR) is a statistical method used to model the relationship between multiple time series variables. It extends the concept of univariate autoregression (AR) to multivariate time series data. In a VAR model, each variable is regressed on its own lagged values as well as the lagged values of all other variables in the system.

Mathematically, a VAR(p) model with $k$ variables can be represented as:

$$
Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + \ldots + A_p Y_{t-p} + \varepsilon_t
$$

Where:
- $Y_t$ is a $k$ -dimensional vector of variables at time $t$.
- $c$ is a $k$-dimensional vector of intercepts.
- $A_1, A_2, \ldots, A_p$ are $k \times k$ coefficient matrices capturing the lagged effects up to lag $p$.
- $\varepsilon_t$ is a $k$-dimensional vector of error terms assumed to be multivariate normally distributed with mean zero and covariance matrix $\Sigma$.

Each coefficient matrix $A_i$ represents the contemporaneous relationship between the variables. For instance, in a VAR(2) model, $A_1$ captures the immediate effect of each variable on itself and on other variables, while $A_2$ captures the effect after one time period.

Here's an example of a bivariate VAR(1) model with two variables:

$$
\begin{bmatrix} 
y_{1,t} \\ 
y_{2,t} 
\end{bmatrix} = 
\begin{bmatrix} 
c_1 \\ 
c_2 
\end{bmatrix} +
\begin{bmatrix} 
a_{11} & a_{12} \\ 
a_{21} & a_{22} 
\end{bmatrix}
\begin{bmatrix} 
y_{1,t-1} \\ 
y_{2,t-1} 
\end{bmatrix} +
\begin{bmatrix} 
\varepsilon_{1,t} \\ 
\varepsilon_{2,t} 
\end{bmatrix}
$$

Where:
- $y_{1,t}$ and $y_{2,t}$ are the variables of interest at time $t$.
- $c_1$ and $c_2$ are intercepts.
- $a_{11}$, $a_{12}$, $a_{21}$, and $a_{22}$  are coefficients capturing the contemporaneous relationships between the variables.
- $\varepsilon_{1,t}$ and $\varepsilon_{2,t}$ are error terms.

Estimation of VAR models typically involves techniques such as least squares or maximum likelihood estimation. These models are commonly used in economics, finance, and other fields to analyze the dynamic interactions between multiple time series variables.


In time series analysis, VAR models are used to capture the dynamic interactions among multiple time series variables. They are particularly useful when you have several interrelated variables and you want to understand how changes in one variable affect the others over time. Here's how you can use `statsmodels`'s `tsa.api.VAR` to model your time series data with three features and tune your model:

1. **Prepare Your Data**: Ensure that your data is in a suitable format. For VAR modeling, you typically need a pandas DataFrame where each column represents a different time series variable.

2. **Split Your Data**: Split your data into training and testing sets. It's common practice to train the model on historical data and then evaluate its performance on unseen data.

3. **Instantiate the VAR Model**: Use `VAR` from `statsmodels.tsa.api` to create an instance of the VAR model. Specify the order of the VAR model (`p`) based on your understanding of the data and potentially using methods like information criteria (e.g., AIC, BIC) or cross-validation to find the optimal order.

4. **Fit the Model**: Fit the VAR model to your training data using the `fit()` method.

5. **Model Diagnosis**: Check the model's diagnostics to ensure that it meets the assumptions of the VAR model. This includes assessing the residuals for autocorrelation, heteroscedasticity, and normality.

6. **Forecasting**: After fitting the model, you can use it to make forecasts for future time points using the `forecast()` method.

7. **Evaluate the Model**: Evaluate the performance of your model using appropriate metrics, such as mean squared error (MSE) or mean absolute error (MAE), on your test dataset.

8. **Tuning the Model**: You can tune your VAR model by adjusting its order (`p`). This can be done by comparing the performance of the model with different orders on a validation dataset or using information criteria like AIC or BIC. You may also experiment with different lag lengths to see which captures the dynamics of your data best.

Here's a basic example of how you can do this using `statsmodels`:

```python
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Assuming df is your pandas DataFrame with three features: 'feature1', 'feature2', 'feature3'

# 1. Prepare your data

# 2. Split your data

# 3. Instantiate the VAR Model
model = VAR(train_data)

# 4. Fit the Model
results = model.fit(maxlags=3)  # Example: using a maximum lag of 3

# 5. Model Diagnosis
print(results.summary())

# 6. Forecasting
forecast = results.forecast(train_data.values, steps=len(test_data))

# 7. Evaluate the Model
mse = mean_squared_error(test_data, forecast)

# 8. Tuning the Model
# You can experiment with different lag orders and assess their performance
```

Remember, tuning the model involves experimentation and validation. You should try different approaches and assess the model's performance based on your specific requirements and domain knowledge.

## SVAR structure selection:

To run unsupervised learning on your energy data to identify clusters, you can use techniques such as K-means clustering or Gaussian mixture models (GMMs). Here's a general approach using Python and popular machine learning libraries like scikit-learn:

1. **Data Preparation**:
   - Prepare your energy data in a suitable format. If you have multiple energy terms over time, you may need to reshape or preprocess the data as necessary.

2. **Feature Selection**:
   - If you have multiple energy terms, decide which features you want to use for clustering. You can use all energy terms or select specific ones based on your analysis goals.

3. **Normalization**:
   - Normalize your data if the energy terms are on different scales. This step ensures that all features contribute equally to the clustering process.

4. **Choose the Algorithm**:
   - Select an unsupervised learning algorithm suitable for clustering. K-means clustering and Gaussian mixture models (GMMs) are common choices for this task.

5. **Model Training**:
   - Train the chosen clustering algorithm on your energy data. Specify the number of clusters (`k` for K-means or the number of components for GMMs) based on your domain knowledge or using techniques like the elbow method or silhouette score.

6. **Cluster Visualization**:
   - Visualize the clusters in your energy data to understand their distribution and relationships. Depending on the dimensionality of your data, you may need to use dimensionality reduction techniques like PCA or t-SNE for visualization.

7. **Interpretation and Analysis**:
   - Analyze the obtained clusters to gain insights into the underlying patterns or structures in your energy data. You can examine the characteristics of each cluster and their relevance to your research questions.

Here's a code example using scikit-learn to perform K-means clustering on energy data:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming energy_data is your preprocessed energy data (e.g., numpy array)
# Perform normalization if necessary
scaler = StandardScaler()
energy_data_normalized = scaler.fit_transform(energy_data)

# Choose the number of clusters (k)
k = 3

# Initialize KMeans model and fit to data
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(energy_data_normalized)

# Predict cluster labels
cluster_labels = kmeans.labels_

# Visualize clusters (for 2D or 3D data)
# Example for 2D data
plt.scatter(energy_data[:, 0], energy_data[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Energy Feature 1')
plt.ylabel('Energy Feature 2')
plt.title('K-means Clustering')
plt.colorbar(label='Cluster')
plt.show()
```

In this code:
- `energy_data` represents your preprocessed energy data.
- The data is normalized using `StandardScaler`.
- K-means clustering is performed with `n_clusters=k`.
- Cluster labels are predicted using `kmeans.labels_`.
- The clusters are visualized in a scatter plot.

You can adapt this example to your specific dataset and analysis requirements. Additionally, consider experimenting with different clustering algorithms and parameters to find the most suitable approach for your data.


python martinize.py -f IgE_bent_ori.pdb -o IgE_bent_ori.top -x CG_IgE_bent_ori.pdb -dssp /usr/local/bin/dssp -p backbone -ff martini22 -ef 500 -el 0.5 -eu 0.9 -ea 0 -ep 0 -elastic -name IgE -cys 0.5 -merge A,B,C,D



Certainly! Here's an example code using Gaussian Mixture Models (GMMs) for clustering your energy data:

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming energy_data is your preprocessed energy data (e.g., numpy array)
# Perform normalization if necessary
scaler = StandardScaler()
energy_data_normalized = scaler.fit_transform(energy_data)

# Choose the number of components (clusters) for GMM
n_components = 3

# Initialize GMM model and fit to data
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(energy_data_normalized)

# Predict cluster labels
cluster_labels = gmm.predict(energy_data_normalized)

# Visualize clusters (for 2D or 3D data)
# Example for 2D data
plt.scatter(energy_data[:, 0], energy_data[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Energy Feature 1')
plt.ylabel('Energy Feature 2')
plt.title('Gaussian Mixture Model Clustering')
plt.colorbar(label='Cluster')
plt.show()
```

In this code:
- `energy_data` represents your preprocessed energy data.
- The data is normalized using `StandardScaler`.
- Gaussian Mixture Model (GMM) clustering is performed with `n_components`.
- Cluster labels are predicted using `gmm.predict()`.
- The clusters are visualized in a scatter plot.

Just like with K-means, you can adapt this example to your specific dataset and analysis requirements. Experiment with different numbers of components (`n_components`) to find the optimal number of clusters for your data. Additionally, consider evaluating the silhouette score or other clustering metrics to assess the quality of the clusters obtained by the GMM model.
