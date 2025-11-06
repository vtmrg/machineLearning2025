# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:15:39 2024

@author: SAMSUNG
"""

#from sklearn import datasets
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB


#%% Chargement de donnés
X = pd.read_excel("Sujet2.xlsx", sheet_name=0, header=0, index_col=0)
print(X.shape)  # ou [n,p]=X.shape ou n=X.shape[0]

#%% Retirer la première colonne
X_C = X.drop(columns=['delta'])
print(X_C.shape)

#%% Analyse Linéaire Discriminante
adl = LinearDiscriminantAnalysis()
Xadl = adl.fit_transform(X_C.iloc[:, 0:5], X_C['phase'])
print(Xadl.shape)  # ou [n,p]=X.shape ou n=X.shape[0]

#%% Faire le plot
# Tracer les données dans le plan des deux vecteurs discriminants
colors = ['turquoise', 'purple', 'pink', 'yellow']

# Assumons que target_names fait référence à un mapping des phases
target_names = ['Phase 0', 'Phase 2', 'Phase 4', 'Phase 5']
phases = [0, 2, 4, 5]  # Correspond aux phases de sommeil

plt.figure(figsize=(8, 6))
for i, phase in enumerate(phases):
    plt.scatter(Xadl[X_C['phase'] == phase, 0], Xadl[X_C['phase'] == phase, 1], color=colors[i], label=target_names[i], s=50, alpha=0.7)

plt.legend(title="Phases de sommeil")
plt.xlabel('1er Discriminant')
plt.ylabel('2e Discriminant')
plt.title('Séparation des phases de sommeil via ALD')
plt.grid(True)
plt.show()

#%% Afficher la variance expliquée par chaque discriminant
print(f"Variance expliquée par les discriminants : {adl.explained_variance_ratio_}")

#%% Graphique de cammenbert

# Calculer les phases uniques et leur fréquence
X_prct = np.unique(X_C['phase'], return_counts=True)

# Création du graphique circulaire (pie chart)
labels = ['Phase 0', 'Phase 2', 'Phase 4', 'Phase 5']  # Noms des phases correspondantes
sizes = X_prct[1]  # Fréquences des phases

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['turquoise', 'purple', 'pink', 'yellow'])
plt.axis('equal')  # Assure que le pie chart est bien circulaire
plt.title('Répartition des phases de sommeil')
plt.show()

#%% Histogramme des phases de sommeil
plt.figure(figsize=(6, 4))
plt.bar(labels, sizes, color=['turquoise', 'purple', 'pink', 'yellow'])
plt.title('Répartition des phases de sommeil')
plt.xlabel('Phases')
plt.ylabel('Fréquence')
plt.show()


#%% Quadratic Discriminant Analysis (QDA) : Covariance différente
# Créer le classificateur QDA
qda = QuadraticDiscriminantAnalysis()

# Entraîner le modèle sur les données d'apprentissage
qda.fit(X_C.iloc[:, 0:5], X_C['phase'])

# Prédire les phases de sommeil
y_pred_qda = qda.predict(X_C.iloc[:, 0:5])

# Calculer et afficher la matrice de confusion
cm_qda = confusion_matrix(X_C['phase'], y_pred_qda)
print("Matrice de confusion pour QDA :")
print(cm_qda)

# Calculer et afficher l'accuracy
accuracy_qda = accuracy_score(X_C['phase'], y_pred_qda)
print(f"Accuracy du QDA : {accuracy_qda}")

#%% Linear Discriminant Analysis (LDA) : Covariance identique
# Utiliser le modèle LDA déjà entraîné
y_pred_lda = adl.predict(X_C.iloc[:, 0:5])

# Calculer et afficher la matrice de confusion pour LDA
cm_lda = confusion_matrix(X_C['phase'], y_pred_lda)
print("Matrice de confusion pour LDA :")
print(cm_lda)

# Calculer et afficher l'accuracy pour LDA
accuracy_lda = accuracy_score(X_C['phase'], y_pred_lda)
print(f"Accuracy du LDA : {accuracy_lda}")

#%% Naïve Bayes Gaussien : Covariance diagonale 
# Créer le classificateur Naïve Bayes Gaussien
gnb = GaussianNB()

# Entraîner le modèle
gnb.fit(X_C.iloc[:, 0:5], X_C['phase'])

# Prédire les phases de sommeil
y_pred_gnb = gnb.predict(X_C.iloc[:, 0:5])

# Calculer et afficher la matrice de confusion pour Naïve Bayes
cm_gnb = confusion_matrix(X_C['phase'], y_pred_gnb)
print("Matrice de confusion pour Naïve Bayes Gaussien :")
print(cm_gnb)

# Calculer et afficher l'accuracy pour Naïve Bayes
accuracy_gnb = accuracy_score(X_C['phase'], y_pred_gnb)
print(f"Accuracy du Naïve Bayes : {accuracy_gnb}")
