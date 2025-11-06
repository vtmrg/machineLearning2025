# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:49:26 2024

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
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score, cross_val_predict 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%% Get dataset

plt.close('all')
sujet_1 = pd.read_excel("Sujet1_artefact.xlsx",sheet_name=0,header=0,index_col=0)
[lines,columns] = sujet_1.shape 
sujet_1.info()

sujet_1_phase = sujet_1['phase']
# Show plot
plt.show()

# Get only the desired columns
sujet_1_util = sujet_1.drop(['beta'], axis = 1)
sujet_1_util.info()

[lines,columns] = sujet_1_util.shape 

#%% Hipnogram

plt.figure(figsize=(10, 5))
plt.plot(sujet_1_phase)
plt.title('Hipnogram for Subject 1')
plt.xlabel('Time interval')
plt.ylabel('Sleep levels')
plt.grid(True)
plt.show()

#%% Réaliser une analyse bivariée (ADL)

# Initialize Linear Discriminant Analysis model
adl = LinearDiscriminantAnalysis()

# Plotting scatter matrix
pd.plotting.scatter_matrix(sujet_1_util.drop(['phase'], axis = 1), c=sujet_1_util['phase'])

# Get the current figure
fig = plt.gcf()
# Add a title
fig.suptitle('Matrice des correlations des variables quantitatives', fontsize=10)
plt.show()

# Fit the model and transform the data
sujet1_transformee = adl.fit_transform(sujet_1_util.iloc[:,0:5], sujet_1_util['phase'])

#%% Représentation dans le plan des 2 vecteurs discriminants

plt.figure()
for label in np.unique(sujet_1_util['phase']):
    plt.scatter(sujet1_transformee[sujet_1_util['phase'] == label][:, 0], 
                sujet1_transformee[sujet_1_util['phase'] == label][:, 1], 
                label=label)
plt.xlabel('Première composante discriminante')
plt.ylabel('Deuxième composante discriminante')
plt.title('LDA of Sujet1 dataset')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

#%% Separate phases of sleep

values,effectif = np.unique(sujet_1_util['phase'], return_counts=True)

plt.figure()
plt.title('Sujet 1')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('Phase of the Sleep')
plt.ylabel('Effectif of each phase')
plt.bar(values, effectif, width=0.8, align='center', color=['blue', 'orange', 'green', 'red'], label=None)
plt.show()

#%% Création de la base d'apprentissage

l = 420  # nombre d'exemples par classe
C1 = sujet_1_util.iloc[:l,:]
C1.insert(6, "etiquette", 0, True)

C2 = sujet_1_util.iloc[l+1:,:]
C2.insert(6, "etiquette", 1, True)

x_train = C1[['delta', 'theta', 'alpha','sigma','puissance']]			
y_train = C1['phase']

x_test = C2[['delta', 'theta', 'alpha','sigma','puissance']]
y_test = C2['phase']

#%% Linear Classifier (LDA)

lin = LinearDiscriminantAnalysis()
y_pred_lin = lin.fit(x_train, y_train)
y_pred_lin = y_pred_lin.predict(x_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (x_test.shape[0], (y_test != y_pred_lin).sum()))

acc_score = balanced_accuracy_score(y_test, y_pred_lin)
ic_lin = [acc_score-1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test))),
                acc_score+1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test)))]
conf_mat_lin = confusion_matrix(y_test, y_pred_lin)
print("LDA Weighted Accuracy IC:", ic_lin)
print("LDA Weighted Accuracy:", acc_score)
print("LDA Confusion Matrix:\n", conf_mat_lin)

#%% Quadratic Classifier (QDA)

quad = QuadraticDiscriminantAnalysis()
y_pred_quad = quad.fit(x_train, y_train)
y_pred_quad = y_pred_quad.predict(x_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (x_test.shape[0], (y_test != y_pred_quad).sum()))

acc_score = balanced_accuracy_score(y_test, y_pred_quad)
ic_quad = [acc_score-1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test))),
                acc_score+1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test)))]
conf_mat_quad = confusion_matrix(y_test, y_pred_quad)
print("QDA Weighted Accuracy IC:", ic_quad)
print("QDA Weighted Accuracy:", acc_score)
print("QDA Confusion Matrix:\n", conf_mat_quad)

#%% Naive Bayes Classifier

gnb = GaussianNB()
y_pred_gnb = gnb.fit(x_train, y_train)
y_pred_gnb = y_pred_gnb.predict(x_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (x_test.shape[0], (y_test != y_pred_gnb).sum()))

acc_score = balanced_accuracy_score(y_test, y_pred_gnb)
ic_gnb = [acc_score-1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test))),
                acc_score+1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test)))]
conf_mat_gnb = confusion_matrix(y_test, y_pred_gnb)
print("Naive Bayes Weighted Accuracy IC:", ic_gnb)
print("Naive Bayes Weighted Accuracy:", acc_score)
print("Naive Bayes Confusion Matrix:\n", conf_mat_gnb)

#%% kppv
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
plt.figure()
# Plot cross-validation scores
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('k-NN Varying k')
plt.show()

# Train the k-NN model with the optimal k
kppv = KNeighborsClassifier(n_neighbors=optimal_k)

y_pred_kppv = kppv.fit(x_train, y_train)
y_pred_kppv = y_pred_kppv.predict(x_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (x_test.shape[0], (y_test != y_pred_kppv).sum()))

acc_score = balanced_accuracy_score(y_test, y_pred_kppv)
ic_kppv = [acc_score-1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test))),
                acc_score+1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test)))]
conf_mat_kppv = confusion_matrix(y_test, y_pred_gnb)
print("Kppv Weighted Accuracy IC:", ic_kppv)
print("Kppv Bayes Weighted Accuracy:", acc_score)
print("Kppv Bayes Confusion Matrix:\n", conf_mat_kppv)


#%% Confusion Matrix Visualization

def plot_confusion_matrix(conf_mat, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot confusion matrices for each classifier
plot_confusion_matrix(conf_mat_lin, title="LDA Confusion Matrix")
plot_confusion_matrix(conf_mat_quad, title="QDA Confusion Matrix")
plot_confusion_matrix(conf_mat_gnb, title="Naive Bayes Confusion Matrix")
plot_confusion_matrix(conf_mat_kppv, title="Kppv Confusion Matrix")


#%% Introduce Reject Class Based on Ambiguity (Low Confidence)

Cr_max =1-(1/4)
Cr = 0.5  # Define an ambiguity threshold
proba_quad = quad.predict_proba(x_test)

# If the highest predicted probability for a sample is below the threshold, classify it as a 'reject'
# Use -1 to represent the 'reject' class numerically
y_pred_reject = np.where(proba_quad.max(axis=1) < Cr, -1, y_pred_quad)

# Now, drop all values where y_pred_reject == -1
# Get indices of non-rejected values
non_rejected_indices = np.where(y_pred_reject != -1)[0]

# Filter the test labels and predictions based on non-rejected indices
y_test_filtered = y_test.iloc[non_rejected_indices]
y_pred_filtered = y_pred_reject[non_rejected_indices]

# Now, you can compute the confusion matrix only for non-rejected values
conf_mat_filtered = confusion_matrix(y_test_filtered, y_pred_filtered)

# Ensure -1 (reject) is included in the list of labels for the confusion matrix
# all_labels = np.unique(y_test)
# print(all_labels)
all_labels=['0',  '2',  '4',  '5', '-1']


acc_score = balanced_accuracy_score(y_test_filtered, y_pred_filtered)
ic_ambiguity = [acc_score-1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test_filtered))),
                acc_score+1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test_filtered)))]
# Confusion matrix with reject class
#conf_mat_reject = confusion_matrix(y_test, y_pred_reject, labels=all_labels)
labels=[0, 2, 4, 5, -1]
conf_mat_density_reject = confusion_matrix(y_test_filtered, y_pred_filtered,labels=[0, 2, 4, 5, -1], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat_density_reject, display_labels=labels)
disp.plot(cmap='Blues', colorbar=True)
print("Ambiguity Weighted Accuracy IC:", ic_ambiguity)
print("Ambiguity Accuracy:", acc_score)


#%% Fonction de calcul de la densité de probabilité

# Calculate the probability density for a given class, using the mean and covariance matrix
def densité_proba(m, co, x):
    # Calculate the inverse and determinant of the covariance matrix
    ci = np.linalg.inv(co)  # inverse of the covariance matrix
    det = np.linalg.det(co)  # determinant of the covariance matrix
    
    # Ensure x and m are column vectors (reshape to match dimensionality of the data)
    x = np.reshape(x, (-1, 1))
    m = np.reshape(m, (-1, 1))
    
    # Calculate the Mahalanobis distance component for the density
    diff = x - m
    exponent = -0.5 * diff.T @ ci @ diff
    
    # Compute the probability density function
    p = np.exp(exponent) / np.sqrt((2 * np.pi) ** len(x) * det)
    
    return p



#%% Distance-based reject function using probability density

def reject_based_on_density(x_train, x_test, m0, co0, m2, co2, m4, co4, m5, co5):
    
    rejections = []
    i=0
    P=np.zeros((len(x_test)))

    for x in x_test.values:
        # Calculate densities for all four classes
        p0 = densité_proba(m0, co0, x)
        p2 = densité_proba(m2, co2, x)
        p4 = densité_proba(m4, co4, x)
        p5 = densité_proba(m5, co5, x)
        
        # Total density and rejection based on threshold
        p_total = p0 + p2 + p4 + p5
        
        P[i]=p_total
        i+=1
        
        Cd = np.percentile(P, 10)
        # Cd = min(P)
        
    for x in x_test.values:
        # Calculate densities for all four classes
        p0 = densité_proba(m0, co0, x)
        p2 = densité_proba(m2, co2, x)
        p4 = densité_proba(m4, co4, x)
        p5 = densité_proba(m5, co5, x)
        
        # Total density and rejection based on threshold
        p_total = p0 + p2 + p4 + p5

        if p_total < Cd:
            rejections.append(-1)  # Reject
        else:
            # Assign to class with highest probability
            if p0 > p2 and p0 > p4 and p0 > p5:
                rejections.append(0)
            elif p2 > p0 and p2 > p4 and p2 > p5:
                rejections.append(2)
            elif p4 > p0 and p4 > p2 and p4 > p5:
                rejections.append(4)
            else:
                rejections.append(5)
    
    return np.array(rejections)

#%% Applying density-based rejection

# Filter data for each class
C0 = x_test[y_test == 0]
C2 = x_test[y_test == 2]
C4 = x_test[y_test == 4]
C5 = x_test[y_test == 5]

# Extract only the features of interest
features = ['delta', 'theta', 'alpha', 'sigma', 'puissance']

# Calculate means for each class
m0 = C0[features].mean().values
m2 = C2[features].mean().values
m4 = C4[features].mean().values
m5 = C5[features].mean().values

# Calculate covariance matrices for each class
co0 = C0[features].cov().values
co2 = C2[features].cov().values
co4 = C4[features].cov().values
co5 = C5[features].cov().values

# Call the rejection function with all classes
rejections = reject_based_on_density(x_train, x_test, m0, co0, m2, co2, m4, co4, m5, co5)

# Update the predicted labels based on rejections (-1 = reject, otherwise use original prediction)
y_pred_density_reject = np.where(rejections == -1, -1, y_pred_lin)

# Now, drop all values where y_pred_reject == -1
# Get indices of non-rejected values
non_rejected_indices = np.where(y_pred_density_reject != -1)[0]

# Filter the test labels and predictions based on non-rejected indices
y_test_filtered = y_test.iloc[non_rejected_indices]
y_pred_filtered = y_pred_reject[non_rejected_indices]

# Now, you can compute the confusion matrix only for non-rejected values
conf_mat_filtered = confusion_matrix(y_test_filtered, y_pred_filtered)

acc_score = balanced_accuracy_score(y_test_filtered, y_pred_filtered)
ic_ambiguity = [acc_score-1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test_filtered))),
                acc_score+1.96*(math.sqrt((acc_score*(1-acc_score))/len(y_test_filtered)))]

# Ensure -1 (reject) is included in the list of labels for the confusion matrix
all_labels = np.unique(np.concatenate([y_test, [-1]]))

# Confusion matrix with density-based rejection
labels=['0', '2', '4', '5','-1']
conf_mat_density_reject = confusion_matrix(y_test, y_pred_density_reject, labels=[0, 2, 4, 5, -1], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat_density_reject, display_labels=labels)
disp.plot(cmap='Blues', colorbar=True)

# print(f"Density-Based Weighted Accuracy IC: {ic_ambiguity:.4f}")
print("Density-Based Accuracy:", acc_score)

