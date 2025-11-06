# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:56:51 2024

@author: victo
"""


#%% Import
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.signal.windows as scw
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score, cross_val_predict 

#%% Get dataset

plt.close('all')
sujet_1 = pd.read_excel("Sujet1.xlsx",sheet_name=0,header=0,index_col=0)
sujet_2 = pd.read_excel("Sujet2.xlsx",sheet_name=0,header=0,index_col=0)
sujet_3 = pd.read_excel("Sujet3.xlsx",sheet_name=0,header=0,index_col=0)

# Get only the desired columns
sujet_1_util = sujet_1.drop(['beta'], axis = 1)
sujet_1_util.info()
sujet_2_util = sujet_2.drop(['beta'], axis = 1)
sujet_2_util.info()
sujet_3_util = sujet_3.drop(['beta'], axis = 1)
sujet_3_util.info()


#%% Creating the data base
subjects = (sujet_1_util, sujet_2_util, sujet_3_util)
all_subjects = np.concatenate(subjects)


x = all_subjects[:,:5]		
y = all_subjects[:,5]

#%%Triage aleatoire par validation croisée

# Set up the scoring metrics
scoring = {
    'balanced_accuracy': 'balanced_accuracy',
    'precision': 'precision_macro',

}
#%% Linear Classifier (LDA)

lin = LinearDiscriminantAnalysis()

# Cross-validate Linear Discriminant Analysis (LDA)
lda_results = cross_validate(lin, x, y, cv=9, scoring=scoring, return_train_score=True)

# Print results for LDA
print("LDA Results:")
print("Weighted Test Accuracy: ", lda_results['test_balanced_accuracy'].mean())
print("Test Precision: ", lda_results['test_precision'].mean())

#%% Quadratic Classifier (QDA)

quad = QuadraticDiscriminantAnalysis()

# Cross-validate Quadratic Discriminant Analysis (QDA)
qda_results = cross_validate(quad, x, y, cv=9, scoring=scoring, return_train_score=True)

# Print results for QDA
print("\nQDA Results:")
print("Weighted Test Accuracy: ", qda_results['test_balanced_accuracy'].mean())
print("Test Precision: ", qda_results['test_precision'].mean())

#%% Naive Bayes Classifier

gnb = GaussianNB()

gnb_results = cross_validate(gnb, x, y, cv=9, scoring=scoring, return_train_score=True)

# Print results for QDA
print("\nNaive Bayes Results:")
print("Weighted Test Accuracy: ", gnb_results['test_balanced_accuracy'].mean())
print("Test Precision: ", gnb_results['test_precision'].mean())

#%% kppv
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=420, shuffle=True)

# Standardize the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Find the optimal value for k using cross-validation
k_values = range(1, 31)  # We will test values from 1 to 30
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Determine the best k
optimal_k = k_values[np.argmax(cv_scores)]
print(f"The optimal value of k is {optimal_k}")

# Plot cross-validation scores
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('k-NN Varying k')
plt.show()

# Train the k-NN model with the optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)

knn_results = cross_validate(knn, x, y, cv=9, scoring=scoring, return_train_score=True)

# Print results for QDA
print("\nKppv Results:")
print("Weighted Test Accuracy: ", knn_results['test_balanced_accuracy'].mean())
print("Test Precision: ", knn_results['test_precision'].mean())

