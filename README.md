# kNN-from-scratch
implements a kNN model and tests ranging values of k to best classify human activity

Using data from smartphones, we can make predictions about what physical action a person was taking based on their movement pattern. The goal of this project is to be able to predict which movement was made using a simple kNN classification, as well as test different values of our hyperparameter, K. Data used in this project can be downloaded from this link: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

This is a large dataset that has already been split into test and training sets. For the purposes on this project, we want to demonstrate the effect that our k parameter has on classification accuracy. This means that we want to shuffle and split that data for each iteration of test. The code in the attached python file concatenates the labels to the attributes as well as training to test sets to create one large dataset that gets resampled from. 
