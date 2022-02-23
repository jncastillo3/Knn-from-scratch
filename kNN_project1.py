#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 12:01:01 2022
@author: nicolecastillo
"""

# prelims
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt

############################
# kNN algorithm functions 
class kNN:
    def __init__(self, data):
        self.train, self.test = train_test_split(shuffle(data), test_size=0.20)

# This function normalizes the data. It then returns training and test subsets 
# with attributes and labels as separate variables
    def norm_data(self):
        for t_data in [self.train, self.test]:
            for j in range(561):
                max_val = t_data[:,j].max()
                min_val = t_data[:,j].min()
                for i in range(len(t_data)):
                    t_data[i,j] = round((t_data[i,j] - min_val)/(max_val-min_val),3)   
        self.X_train = self.train[:,0:561]
        self.y_train = self.train[:,-1]
        self.X_test = self.test[:,0:561]
        self.y_test = self.test[:,-1]
        return self.X_train, self.y_train, self.X_test, self.y_test
        
# This function makes a prediction of class. it calcualtes the euclidean distance, and then
# sorts the distances in order of the current instance being predicted. It then only considers
# the k nearest instances, looks at their labels, and returns the mode
    def predict_new_instance(self, new_data, k):
        euclidean_dist = cdist(new_data, self.X_train) #euclidean distance; each test instance to each training instance 
        predicted_labels = []
        for i in range(new_data.shape[0]): #for each test input, sort distances from i and get index of closest points based on k 
            nearest_index = np.argsort(euclidean_dist[i])[:k] 
            nearest_labels = self.y_train[nearest_index] 
            nearest_labels.sort()
            predicted_labels.append(stats.mode(nearest_labels)[0][0])
        return np.array(predicted_labels)

# This functions determines the accuracy of the algorithm. given some data to classify, and
# their labels, it compares the predicted labels to the actual labels.
    def kNN_accuracy(self, new_data, data_labels, k):
        predicted_labels = self.predict_new_instance(new_data, k)
        prediction_accuracy = np.mean(predicted_labels == data_labels)
        return prediction_accuracy

############################

# Import the relevant data
X_train = np.loadtxt('/Users/nicolecastillo/Downloads/UCI HAR Dataset/train/X_train.txt', dtype=float)
X_test = np.loadtxt('/Users/nicolecastillo/Downloads/UCI HAR Dataset/test/X_test.txt', dtype=float)
y_train = np.loadtxt('/Users/nicolecastillo/Downloads/UCI HAR Dataset/train/y_train.txt', dtype=float)
y_test = np.loadtxt('/Users/nicolecastillo/Downloads/UCI HAR Dataset/test/y_test.txt', dtype=float)

# Data for this dataset comes split into a training and test set already but we want to be able to
# test for best values of k, so we want to shuffle the data and split repeatedly. So, lets concatenate.
train_w_labels = np.insert(X_train, 561, y_train, axis=1)
test_w_labels = np.insert(X_test, 561, y_test, axis=1)
full_data =np.vstack((train_w_labels,test_w_labels))

# compute the accuracy of the k-NN model when used to make predictions for instances in the training set
# this will loop through the algorith 20 times and return the accuracy of each iteration.
accuracy_by_k = []
for i in range(1,20,2):
    for j in range(10):
        if __name__=="__main__":
            knn_model = kNN(full_data)
            return_normed_data = knn_model.norm_data()
            predictions = knn_model.predict_new_instance(return_normed_data[0], 10)
            accuracy = knn_model.kNN_accuracy(return_normed_data[0], return_normed_data[1], 10)
            accuracy_by_k.append([i , accuracy])

# Get standard deviation and mean by value of k - this is for the plots
acc_df = pd.DataFrame(accuracy_by_k)
sd_by_k = acc_df.groupby(0)[1].std()
mean_by_k = acc_df.groupby(0)[1].mean()

# make plot
plt.plot(np.unique(acc_df.iloc[:,0]), mean_by_k, 'o-g')
plt.errorbar(np.unique(acc_df.iloc[:,0]), mean_by_k, yerr = sd_by_k, ecolor = 'g')
plt.title('Average kNN accuracy on TRAINING data')
plt.ylabel('classifier Accuracy (20 iterations)')
plt.xlabel('size of k')
    
# compute the accuracy of the k-NN model when used to make predictions for instances in the testing set
# Same as before, but not for the test data
accuracy_by_k = []
for i in range(1,53,2):
    for j in range(10):
        if __name__=="__main__":
            knn_model = kNN(full_data)
            return_normed_data = knn_model.norm_data()
            predictions = knn_model.predict_new_instance(return_normed_data[2], i)
            accuracy = knn_model.kNN_accuracy(return_normed_data[2], return_normed_data[3], i)
            accuracy_by_k.append([i , accuracy])   

# Get standard deviation and mean by value of k
acc_df = pd.DataFrame(accuracy_by_k)
sd_by_k = acc_df.groupby(0)[1].std()
mean_by_k = acc_df.groupby(0)[1].mean()
       
#make plot
plt.plot(np.unique(acc_df.iloc[:,0]), mean_by_k, 'o-g')
plt.errorbar(np.unique(acc_df.iloc[:,0]), mean_by_k, yerr = sd_by_k, ecolor = 'g')
plt.title('Average kNN accuracy on TEST data')
plt.ylabel('Classifier accuracy (20 iterations)')
plt.xlabel('size of k')