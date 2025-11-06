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

