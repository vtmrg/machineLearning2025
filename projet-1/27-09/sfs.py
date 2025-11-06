# -*- coding: utf-8 -*-
"""
Sélection de caractéristiques pertinentes avec Sequential Forward Search (SFS)
"""

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np

#%% Charger les données du sujet sélectionné
X_sujet = pd.read_excel("sujet1.xlsx", sheet_name=0, header=0, index_col=0)  # Ajuster le fichier selon le sujet choisi

# Préparer les données (sans la colonne 'delta' et la colonne 'phase')
X = X_sujet.drop(columns=['delta', 'phase'])  # Retirer 'delta' et 'phase'
y = X_sujet['phase']

#%% Sélection du classificateur (QDA dans cet exemple)
qda = QuadraticDiscriminantAnalysis()

#%% Appliquer Sequential Forward Search (SFS) avec validation croisée
# Créer le sélecteur de caractéristiques
sfs = SequentialFeatureSelector(qda, n_features_to_select=3, direction='forward', cv=5)

# Ajuster le sélecteur sur les données
sfs.fit(X, y)

# Obtenir les caractéristiques sélectionnées
selected_features = sfs.get_support()

# Afficher les caractéristiques sélectionnées
print(f"Caractéristiques sélectionnées : {X.columns[selected_features]}")

#%% Évaluer la performance avec les caractéristiques sélectionnées
# Créer un nouveau jeu de données avec les caractéristiques sélectionnées
X_selected = X.loc[:, selected_features]

# Effectuer une validation croisée avec les caractéristiques sélectionnées
scores = cross_val_score(qda, X_selected, y, cv=5)

# Afficher les résultats de la validation croisée
print(f"Scores de validation croisée avec les caractéristiques sélectionnées : {scores}")
print(f"Score moyen : {np.mean(scores):.4f}")
