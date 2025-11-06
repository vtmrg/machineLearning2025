# -*- coding: utf-8 -*-
"""
Rejet en distance et en ambiguïté pour QDA
"""

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#%% Chargement de données
X = pd.read_excel("\Sujet2.xlsx", sheet_name=0, header=0, index_col=0)
X_C = X.drop(columns=['delta'])

#%% Séparer les données en ensembles d'apprentissage (420 premières lignes) et de test (reste)
X_train = X_C.iloc[:420, 0:5]  # 420 premières lignes pour l'apprentissage
y_train = X_C['phase'][:420]

X_test = X_C.iloc[420:, 0:5]  # Reste pour le test
y_test = X_C['phase'][420:]

#%% Entraîner le modèle QDA sur l'ensemble d'apprentissage
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

#%% Prédire sur l'ensemble de test
y_pred = qda.predict(X_test)
y_proba = qda.predict_proba(X_test)

#%% Calcul des distances de Mahalanobis pour chaque point
def mahalanobis_distance(x, mean, cov):
    cov_inv = np.linalg.inv(cov)  # Inverser la matrice de covariance
    diff = x - mean
    dist = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
    return dist

# Calcul manuel des matrices de covariance et des moyennes pour QDA
classes = qda.classes_
means = qda.means_

# Calcul des covariances pour chaque classe
covariances = []
for i, class_label in enumerate(classes):
    X_class = X_train[y_train == class_label].iloc[:, 0:5]
    covariance = np.cov(X_class, rowvar=False)
    covariances.append(covariance)

# Calcul des distances de Mahalanobis pour chaque point de test
distances = []
for i, row in X_test.iterrows():
    class_distances = []
    for j, class_mean in enumerate(means):
        dist = mahalanobis_distance(row, class_mean, covariances[j])
        class_distances.append(dist)
    distances.append(min(class_distances))  # Distance minimale parmi les classes

#%% Rejet en ambiguïté et en distance
# Définir les seuils
distance_threshold = 10  # Seuil de distance pour le rejet
proba_threshold = 0.7  # Seuil de probabilité pour le rejet en ambiguïté

# Appliquer les rejets sur les prédictions
y_pred_reject = []
for i in range(len(y_pred)):
    # Rejet en distance
    if distances[i] > distance_threshold:
        y_pred_reject.append(-1)
    # Rejet en ambiguïté
    elif max(y_proba[i]) < proba_threshold:
        y_pred_reject.append(-1)
    else:
        y_pred_reject.append(y_pred[i])

#%% Calculer la nouvelle matrice de confusion avec rejet
cm_reject = confusion_matrix(y_test, y_pred_reject, labels=[0, 2, 4, 5, -1])
print("Matrice de confusion avec rejet (distance et ambiguïté) :")
print(cm_reject)

# Calculer l'accuracy et la balanced accuracy avec rejet
accuracy_reject = accuracy_score(y_test, y_pred_reject)
balanced_acc_reject = balanced_accuracy_score(y_test, y_pred_reject, adjusted=True)
print(f"Accuracy avec rejet : {accuracy_reject:.4f}")
print(f"Balanced Accuracy avec rejet : {balanced_acc_reject:.4f}")


class_labels = ['0', '2', '4', '5', '-1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm_reject, display_labels=class_labels)
disp.plot(cmap='Blues',  colorbar=True)