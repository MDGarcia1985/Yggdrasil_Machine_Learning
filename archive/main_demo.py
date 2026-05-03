"""
# main.py
This is the main module for an ML app that identifies whether a bomber or a fighter
is coming in.

Import libs from scikit-learn.org
to make a classifier based on a decision tree.

Written by: Michael Garcia
Date: 03/22/2026
For: CSC373 Machine Learning
"""
from __future__ import annotations

from sklearn import tree

# Make sure to let the user know what is going on
# We will need features and labels
print("Loading model features and labels...")
features = [[42,60],[35,55],[29,35],[23,24],[26,28],[22,25],]
labels = [1,1,1,0,0,0]

#Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#Prediction
print("Model prediction: ")
print("\n\n\n")
# This should be a fighter
print("let's predict a plane that has a wing span of 25 feet and a length of 28 feet")
print(clf.predict([[25,28]]))

# This should be a bomber
print("\n\n\n")
print("let's predict a plane that has a wing span of 38 feet and a length of 43 feet")
print(clf.predict([[38,43]]))

# Notice:
# These sizes were chosen because they are not in the training data