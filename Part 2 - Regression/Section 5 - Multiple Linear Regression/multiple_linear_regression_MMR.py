#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:08:13 2018

@author: moshiur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encode the categorical variable state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap. you should take one dummy variable out of the array
X = X[:, 1:]

# split the datset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
linregressor = LinearRegression()
linregressor.fit(X_train, y_train)

# test performance of the multiple linear regression model
y_pred = linregressor.predict(X_test)
 
# optimal model using Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis=1)

# you can do Backward elimination manually, removing parameters whose
# p values are greater than SL or also use the function backwardElimination() 


# manual Backward elimination
"""
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
linregressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
linregressor_OLS.summary()

# removing the 2nd variable (p value 0.99) and model again
X_opt = X[:, [0, 1, 3, 4, 5]]
linregressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
linregressor_OLS.summary()

# removing the 1st variable (P values 0.940) and model again 
X_opt = X[:, [0, 3, 4, 5]]
linregressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
linregressor_OLS.summary()

# removing the 4th variable (P values 0.602) and model again 
X_opt = X[:, [0, 3, 5]]
linregressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
linregressor_OLS.summary()

# removing the 5th variable (P values 0.602) and model again 
X_opt = X[:, [0, 3]]
linregressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
linregressor_OLS.summary()

# removing the constant variable (P values 0.259) and model again 
X_opt = X[:, [4, 5]]
linregressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
linregressor_OLS.summary()
"""
# Backward elimination using the function backwardElimination() 
# function
def backwardElimination(X, SL):
    numVars = np.size(X, axis=1) # number of parameters along the column of X
    
    for i in range(0, numVars):
        linreg_OLS = sm.OLS(endog=y, exog=X).fit()
        maxPvalue = max(linreg_OLS.pvalues).astype(float) # max p value
        if maxPvalue > SL:
            for j in range (0, numVars - i):
                # remove the element X[:, j] from the array for which pavale > SL
                if linreg_OLS.pvalues[j] == maxPvalue:
                    X = np.delete(X, j, 1)
    linreg_OLS.summary()
    return X

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_modeled = backwardElimination(X_opt, SL)
        
    