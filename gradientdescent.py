#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Linear Regression using Gradient Descent

# Importing the necessary packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Retrieving the URL to the dataset, which can also be found here: https://archive.ics.uci.edu/dataset/9/auto+mpg

# In[2]:
print("\npart1.py\nMatthew Chimitt\nmmc200005")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
col = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin", "Car Name"]


# Getting the dataframe to use

# In[4]:


df = pd.read_csv(url, names=col, delim_whitespace=True)


# In[9]:


print("\nDataframe")
print(df)


# #### Correlation
# Using the Cylinders, displacement, weight, acceleration, and model year for the features of the model.
# Not using car name as it is not a continuous number, but rather a string.

# In[8]:


# print("\nCorrelation:")
# print(df.corr())


# #### Scrambling the data
# Randomizing and then setting the X and y values
# X is all of the data we use to predict y

# In[10]:


# Scramble the Data a bit
df = df.sample(frac=1)
X = df[['Cylinders', 'Displacement', 'Weight', 'Acceleration', 'Model Year']]
y = df['MPG']


# #### Splitting the data
# 80% of the data is training, 20% is testing

# In[11]:


ratio = 0.80
rows = df.shape[0]
train_size = int(rows*ratio)

train_X = X[0:train_size]
train_y = y[0:train_size]
test_X = X[train_size:]
test_y = y[train_size:]


# In[12]:


# Convert the test data from a data frame to numpy array
test_X = test_X.to_numpy()
test_y = test_y.to_numpy()


# In[13]:


# Converting the training data to be a numpy array
X = train_X.to_numpy()
# Transposing X to use in gradient descent
X = X.T

y = train_y.to_numpy()

print(X.shape)
print(y.shape)


# In[14]:


plt.plot(X[0], y, 'ro')
plt.ylabel('MPG (output)')
plt.xlabel('Cylinders')
plt.title('Cylinders vs MPG')
print("Cylinder vs MPG Graph Popup: Close the popup to continue.")
plt.show()

# In[15]:


plt.plot(X[1], y, 'ro')
plt.ylabel('MPG (output)')
plt.xlabel('Displacement')
plt.title('Displacement vs MPG')
print("Displacement vs MPG Graph Popup: Close the popup to continue.")
plt.show()

# In[16]:


plt.plot(X[2], y, 'ro')
plt.ylabel('MPG (output)')
plt.xlabel('Weight')
plt.title('Weight vs MPG')
print("Weight vs MPG Graph Popup: Close the popup to continue.")
plt.show()

# In[17]:


plt.plot(X[3], y, 'ro')
plt.ylabel('MPG (output)')
plt.xlabel('Acceleration')
plt.title('Acceleration vs MPG')
print("Acceleration vs MPG Graph Popup: Close the popup to continue.")
plt.show()

# In[18]:


plt.plot(X[4], y, 'ro')
plt.ylabel('MPG (output)')
plt.xlabel('Model Year')
plt.title('Model Year vs MPG')
print("Model Year vs MPG Graph Popup: Close the popup to continue.")
plt.show()

# ## Gradient Descent
# #### SSR Gradient
# res refers to the residuals
# returns the weights|

# In[19]:

print("Getting Weights...")

def ssr_gradient(x, y, w):  
    res = w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]*x[3] + w[5]*x[4] - y  
    return res.mean(), (res * x[0]).mean(), (res * x[1]).mean(), (res * x[2]).mean(), (res * x[3]).mean(), (res * x[4]).mean()


# performing gradient descent

# In[20]:


def gradient_descent(
     gradient, x, y, start, learn_rate=0.1, n_iter=500, tolerance=1e-10
 ):
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate * np.array(gradient(x, y, vector))
    # print(diff)
    # type(diff)
    if np.all(np.abs(diff) <= tolerance):
      break
    vector += diff
#     type(vector)
  return vector


# Using the gradient descent function and the ssr_gradient to get the best weights

# In[22]:


weights = gradient_descent(
    ssr_gradient, X, y, start=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], learn_rate=0.0000001,
    n_iter=1000000
)
print("\nWeights from Gradient Descent: ")
print(weights)
print("\n")


# #### Doing a small test
# sample is the predicted value on the first row of data from the training set
# the function is w0 + w1x0 + w2x1 + w3x2 + w4x3 + w5x4
# 
# The first value printed is the predicted y and the second value printed is the actual y

# In[23]:


sample = weights[0] + (weights[1]*X[0][0]) + (weights[2] * X[1][0]) + (weights[3] * X[2][0]) + (weights[4] * X[3][0]) + (weights[5] * X[4][0])
print("Comparing a single prediction to the actual value:")
print("Prediction:")
print(sample)
print("Actual:")
print(y[0])

print("\n\nEquation:")
print(str(weights[0]) + " + " + str(weights[1]) + " * Cylinders + " + str(weights[2]) + " * Displacement + " + str(weights[3]) + " * Weight + " + str(weights[4]) + " * Acceleration + " + str(weights[5]) + " * Model Year\n\n") 


# ## Plotting Y Predicted vs Y Actual for the Training Data

# In[24]:


# y predicted for the testing data
y_predicted = weights[0] + weights[1]*test_X.T[0] + weights[2]*test_X.T[1] + weights[3]*test_X.T[2] + weights[4]*test_X.T[3] + weights[5]*test_X.T[4]


# In[25]:


fig, ax = plt.subplots(figsize=[10.,10.], dpi=100)
ax.plot(y_predicted, label="Y Predicted")
ax.plot(test_y, label="Y Actual")
ax.set_title("Comparing the Predicted and Actual Values of the Testing Data")
ax.legend()

plt.tight_layout()
print("Predicted vs Actual Graph Popup: Close the popup to continue.")
plt.show()

# ### Calculating the MSE
# calc_error finds the (actual - predicted)^2 for a given row in the data
# mse sums all of the values found in calc_error, dividing the total by n

# In[26]:


def calc_error(x, y, w, i):
    intercept = w[0]
    theta = w[1:]
    pred = intercept + np.dot(x[i], theta)
    actual = y[i]
    error = (actual - pred) ** 2
    return error

def mse(x, y, w):
    sum = 0
    for i in range(x.shape[0]):
        sum += calc_error(x, y, w, i)
    out = sum/x.shape[0]
    return out


# #### Training and Testing MSE and RMSE

# In[27]:

print("\n\n")
train_error = mse(X.T, y, weights)
test_error = mse(test_X, test_y, weights)
print("Train Error: " + str(train_error))
print("Test Error: " + str(test_error))


# In[28]:


print("\nTrain RMSE: " + str(np.sqrt(train_error)))
print("Test RMSE: " + str(np.sqrt(test_error)))


# #### Getting the R^2 Value

# In[29]:


from sklearn.metrics import r2_score


# In[31]:


y_predicted = weights[0] + weights[1]*X[0] + weights[2]*X[1] + weights[3]*X[2] + weights[4]*X[3] + weights[5]*X[4]
y_predicted
print("\nR^2 Value for Training:")
print(r2_score(y, y_predicted))

y_predicted_test = weights[0] + weights[1]*test_X.T[0] + weights[2]*test_X.T[1] + weights[3]*test_X.T[2] + weights[4]*test_X.T[3] + weights[5]*test_X.T[4]
print("\nR^2 Value for Testing:")
print(r2_score(test_y, y_predicted_test))


# # Answering Question 6 in Part 1
# ### Are you satisfied that you have found the best solution? Explain:
# No I do not believe that I have found the best solution. This is because nothing is the real world tends to be represented in linearly. Because of this, there is no way to find the "best" solution to a question using gradient descent, and linear regression.

# In[ ]:




