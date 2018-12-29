#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 23:21:48 2018

@author: moshiur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""
# as the dataset is very small there's no need to divide it to training
# and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)

# fitting Polynomial Regression ot the dataset
from sklearn.preprocessing import PolynomialFeatures
#create a polynomial version of X
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# create a Linear Regression model and the fit the X_poly
linreg2 = LinearRegression()
linreg2.fit(X_poly, y)

# visualize Linear Regression results
plt.scatter(X, y, color='blue')
plt.plot(X, linreg.predict(X), color='red')
plt.title('Salary Linear Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualize Polynomial Regression results
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.figure()
plt.scatter(X, y, color='blue')
plt.plot(X, linreg2.predict(poly_reg.fit_transform(X)), color='red')
plt.title('Salary Polynomial Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predict with Linear Regression
linreg.predict([[6]])

# predict with Polynomial Regression
linreg2.predict(poly_reg.fit_transform([[6.5]]))