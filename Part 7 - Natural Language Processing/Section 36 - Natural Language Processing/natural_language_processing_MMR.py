#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:29:40 2018

@author: moshiur
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# cleaning the texts
import re
import nltk
# download the stopwords list
nltk.download('stopwords')
# import the downloaded stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
# cleaned Review
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    # convert the review string to list
    review = review.split()
    # stemming: keeping only the root of a word
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    # join the words in review
    review = ' '.join(review)
    corpus.append(review)

# create the bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
# create the sparse matrix X (input for the ML model)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# build a decision tree model

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting classifier to the Training set
from sklearn.tree import  DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)