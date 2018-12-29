#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:52:01 2018

@author: moshiur
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
# convert y vector to a matrix when putting it as an input to the fit_transform
# function. then after transforming, converty matrix-y to vector-y using np.sqeeze
y = np.squeeze(sc_y.fit_transform(y.reshape(-1, 1)))
#y = sc_y.fit_transform(y.reshape(-1, -1))

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X, y)

# Predicting a new result, put the test sample (6.5) as an array i.e., as [[6.5]]
# it is to be noted that [] denotes a vector and [[]] denotes an array
y_pred = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, svr_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()