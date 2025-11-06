# -*- coding: utf-8 -*-
"""
Rejet basé sur la densité et l'ambiguïté avec calcul sur la base d'apprentissage pour QDA
"""

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

#%% Chargement de données
X = pd.read_excel("Sujet1_artefact.xlsx", sheet_name=0, header=0, index_col=0)
X_C = X.drop(columns=['delta'])

#%% Séparer les données en ensembles d'apprentissage (420 premières lignes) et de test (reste)
X_train = X_C.iloc[:420, 0:5]  # 420 premières lignes pour l'apprentissage
y_train = X_C['phase'][:420]

X_test = X_C.iloc[420:, 0:5]  # Reste pour le test
y_test = X_C['phase'][420:]

#%% Entraîner le modèle QDA sur l'ensemble d'apprentissage avec stockage des covariances
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(X_train, y_train)

#%% Prédire sur l'ensemble de test
y_pred = qda.predict(X_test)
y_proba = qda.predict_proba(X_test)

#%% Fonction pour calculer la densité de probabilité multivariée gaussienne
def densite_proba(m, co, x):
    """Calculer la densité de probabilité multivariée gaussienne pour un point x."""
    ci = np.linalg.inv(co)  # Inverser la matrice de covariance
    det = np.sqrt(np.linalg.det(co))  # Calculer le déterminant de la matrice de covariance
    xt = np.reshape(x, (len(x), 1))  # Reshape de x en vecteur colonne
    mt = np.reshape(m, (len(m), 1))  # Reshape de m en vecteur colonne
    
    # Calcul de la densité gaussienne multivariée
    p = np.exp(-0.5 * (xt - mt).T @ ci @ (xt - mt)) / (2 * np.pi * det)
    
    return p[0, 0]  # Retourner la valeur scalaire de la densité

#%% Récupération des covariances stockées par QDA
covariances = qda.covariance_  # Utiliser les covariances calculées automatiquement par QDA
classes = qda.classes_
means = qda.means_  # Moyennes des classes calculées par QDA

#%% Calcul de la densité de probabilité totale sur la base d'apprentissage
# On calcule les densités sur les points d'apprentissage (X_train)
train_densities = np.zeros(len(X_train))  # Stocker la densité totale pour chaque point d'apprentissage

for i, x in enumerate(X_train.values):  # x représente chaque point d'apprentissage
    p = 0  # Initialiser la densité totale à 0
    for j, m in enumerate(means):  # m représente la moyenne de chaque classe
        co = covariances[j]  # La covariance correspondante
        p += densite_proba(m, co, x)  # Calculer et ajouter la densité pour chaque classe
    train_densities[i] = p  # Stocker la densité totale pour le point x d'apprentissage

#%% Définir le seuil de rejet basé sur les densités calculées sur la base d'apprentissage
cd = min(train_densities)  # Seuil basé sur 90% de la densité minimale dans la base d'apprentissage
print(f"Seuil de rejet basé sur la densité d'apprentissage : {cd:.4f}")

#%% Calcul de la densité de probabilité totale sur la base de test
# Calcul de la densité totale sur chaque point de test en utilisant les mêmes moyennes et covariances
test_densities = np.zeros(len(X_test))

for i, x in enumerate(X_test.values):  # x représente chaque point de test
    p = 0  # Initialiser la densité totale à 0
    for j, m in enumerate(means):  # m représente la moyenne de chaque classe
        co = covariances[j]  # La covariance correspondante
        p += densite_proba(m, co, x)  # Calculer et ajouter la densité pour chaque classe
    test_densities[i] = p  # Stocker la densité totale pour le point de test

#%% Appliquer le rejet en fonction de la densité et de l'ambiguïté
y_pred_reject = []
cr = 0.5  # Seuil de rejet en ambiguïté (à ajuster si besoin)

for i in range(len(y_pred)):
    # Densité totale de test inférieure au seuil d'apprentissage -> Rejet en densité
    if test_densities[i] < cd:
        y_pred_reject.append(-1)
    # Vérifier l'ambiguïté : Si la probabilité maximale est inférieure au seuil `cr`
    elif max(y_proba[i]) < cr:
        y_pred_reject.append(-1)
    else:
        y_pred_reject.append(y_pred[i])

#%% Calculer la nouvelle matrice de confusion avec rejet
cm_reject = confusion_matrix(y_test, y_pred_reject, labels=[0, 2, 4, 5, -1])
print("Matrice de confusion avec rejet (densité et ambiguïté) :")
print(cm_reject)

# Calculer l'accuracy et la balanced accuracy avec rejet
accuracy_reject = accuracy_score(y_test, y_pred_reject)
balanced_acc_reject = balanced_accuracy_score(y_test, y_pred_reject, adjusted=True)
print(f"Accuracy avec rejet : {accuracy_reject:.4f}")
print(f"Balanced Accuracy avec rejet : {balanced_acc_reject:.4f}")

#%% Visualisation des densités
# Tracé de l'histogramme des densités sur la base d'apprentissage
plt.figure(figsize=(8, 5))
plt.hist(train_densities, bins=30, color='green', alpha=0.7)
plt.axvline(x=cd, color='red', linestyle='--', label='Seuil de rejet basé sur la densité')
plt.title('Distribution des densités de probabilité (Base d\'apprentissage)')
plt.xlabel('Densité de probabilité totale (Apprentissage)')
plt.ylabel('Fréquence')
plt.legend()
plt.show()

# Tracé de l'histogramme des densités sur la base de test
plt.figure(figsize=(8, 5))
plt.hist(test_densities, bins=30, color='blue', alpha=0.7)
plt.axvline(x=cd, color='red', linestyle='--', label='Seuil de rejet basé sur la densité d\'apprentissage')
plt.title('Distribution des densités de probabilité (Base de test)')
plt.xlabel('Densité de probabilité totale (Test)')
plt.ylabel('Fréquence')
plt.legend()
plt.show()
