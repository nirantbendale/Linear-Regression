#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


# In[2]:


# Load the Boston Housing Prices dataset
boston = load_boston()
data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target


# In[3]:


# Summary statistics of the dataset
print(data.describe())


# In[4]:


# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[5]:


# Histograms of key features
plt.figure(figsize=(15, 10))
features = ['RM', 'LSTAT', 'CRIM', 'PRICE']
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.histplot(data[feature], kde=True, bins=30, color='b')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# In[6]:


# Scatter plots of key features vs. PRICE
plt.figure(figsize=(15, 5))
for i, feature in enumerate(['RM', 'LSTAT', 'CRIM']):
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x=feature, y='PRICE', data=data, alpha=0.7)
    plt.title(f'{feature} vs. PRICE')
plt.tight_layout()
plt.show()


# In[7]:


# Select features and target variable
X = data[['RM', 'LSTAT', 'CRIM']]
y = data['PRICE']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)


# In[9]:


# Make predictions on the test data
y_pred = model.predict(X_test)


# In[10]:


# Calculate Mean Squared Error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients and evaluation metrics
print(f'Linear Regression Coefficients: {model.coef_}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2) Score: {r2:.2f}')


# In[11]:


# Plot the regression line and actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()


# In[ ]:




