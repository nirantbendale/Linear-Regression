# Linear-Regression
This project uses the scikit-learn library to predict housing prices in the Boston area based on various features such as the average number of rooms per dwelling (RM), the percentage of lower-status population (LSTAT), and the crime rate (CRIM). Linear regression is employed as the predictive model.

## Importing the neccesary libraries <br>
`import numpy as np`  # For numerical operations <br>
`import pandas as pd`  # For data manipulation <br>
`import matplotlib.pyplot as plt`  # For data visualization <br>
`from sklearn.datasets import load_boston`  # For loading the dataset <br>
`from sklearn.model_selection import train_test_split`  # For splitting the dataset <br>
`from sklearn.linear_model import LinearRegression`  # For building the Linear Regression model <br>
`from sklearn.metrics import mean_squared_error, r2_score`  # For model evaluation <br>
