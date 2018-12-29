#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 09:41:51 2018

@author: moshiur
"""

# part-1: data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# remove the first column of the country variable to avoid 'dummy variable trap'
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# part-2: building the ANN
# import Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize the ANN
classifier = Sequential()

# add input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# add 2nd hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=50)

# predict on the Test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # adding a threshold of 0.5 for customer churn output

# build the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)