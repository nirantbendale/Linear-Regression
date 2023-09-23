# Linear-Regression
This project uses the scikit-learn library to predict housing prices in the Boston area based on various features such as the average number of rooms per dwelling (RM), the percentage of lower-status population (LSTAT), and the crime rate (CRIM). Linear regression is employed as the predictive model.

### Importing the necessary libraries 
`import numpy as np`  # For numerical operations <br>
`import pandas as pd`  # For data manipulation <br>
`import matplotlib.pyplot as plt`  # For data visualization <br>
`from sklearn.datasets import load_boston`  # For loading the dataset <br>
`from sklearn.model_selection import train_test_split`  # For splitting the dataset <br>
`from sklearn.linear_model import LinearRegression`  # For building the Linear Regression model <br>
`from sklearn.metrics import mean_squared_error, r2_score`  # For model evaluation <br>

### We begin by importing the required libraries, including NumPy for numerical operations, Pandas for data manipulation, Matplotlib for data visualization, and scikit-learn functions for loading data, model building, and evaluation.
`boston = load_boston()` <br>
`data = pd.DataFrame(data=boston.data, columns=boston.feature_names)`<br>
`data['PRICE'] = boston.target`<br>
[['.png]]
### We load the Boston Housing Prices dataset using load_boston() and create a Pandas DataFrame to store the data. We also add a 'PRICE' column to the DataFrame to hold the target variable (housing prices). 

`X = data[['RM', 'LSTAT', 'CRIM']]`  # Select features and target variable  <br>
`y = data['PRICE'] `

### We select three specific features ('RM', 'LSTAT', 'CRIM') as predictors (X) and the 'PRICE' column as the target variable (y).
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)` # Split the dataset into training and testing sets

### We split the dataset into training and testing sets using train_test_split(), with 80% of the data used for training and 20% for testing. The random_state parameter ensures reproducibility.
`model = LinearRegression()`# Create a Linear Regression model  <br>

### We create an instance of the Linear Regression model. 
`model.fit(X_train, y_train)` # Fit the model to the training data <br>

### We fit (train) the model using the training data.
`y_pred = model.predict(X_test)` # Make predictions on the test data <br>

### We use the trained model to make predictions on the test data.
`mse = mean_squared_error(y_test, y_pred)` # Calculate Mean Squared Error  <br>
`r2 = r2_score(y_test, y_pred)` # Calculate R-squared score <br>

### We calculate the Mean Squared Error (MSE) and R-squared (R2) score to evaluate the performance of the model.
`print(f'Linear Regression Coefficients: {model.coef_}')` <br>
`print(f'Mean Squared Error (MSE): {mse:.2f}')` <br>
`print(f'R-squared (R2) Score: {r2:.2f}')` <br>

### We print the coefficients of the Linear Regression model and the evaluation metrics, including MSE and R2 score, to assess how well the model fits the data.
`plt.scatter(y_test, y_pred)` <br>
`plt.xlabel("Actual Prices")` <br>
`plt.ylabel("Predicted Prices")` <br>
`plt.title("Actual Prices vs. Predicted Prices")` <br>
`plt.show()`

### Finally, we create a scatter plot to visualize the relationship between actual and predicted housing prices, helping us understand how well the model's predictions align with the true values. <br> <br>
### This project demonstrates the use of linear regression for predicting housing prices based on specific features and evaluates the model's performance using common regression evaluation metrics.

