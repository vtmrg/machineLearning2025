# -*- coding: utf-8 -*-
"""
Validation inter-sujet avec balanced accuracy et mélange aléatoire des données
"""

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle  # Pour mélanger les données

#%% Fonction pour charger les données de plusieurs sujets
def charger_donnees_sujets(fichiers):
    """Charge les données de plusieurs sujets et retourne les ensembles de données."""
    donnees_sujets = []
    for fichier in fichiers:
        data = pd.read_excel(fichier, sheet_name=0, header=0, index_col=0)
        donnees_sujets.append(data)
    return donnees_sujets

#%% Fonction pour préparer les ensembles d'entraînement et de test inter-sujet
def preparer_ensembles_inter_sujet_random(donnees_sujets, ratio_train=0.7):
    """
    Prépare les ensembles d'apprentissage et de test pour validation inter-sujet
    avec mélange aléatoire des données.
    """
    # Concaténer toutes les bases de données des sujets
    donnees_concatenees = pd.concat(donnees_sujets, axis=0)
    
    # Mélanger les données aléatoirement
    donnees_concatenees = shuffle(donnees_concatenees, random_state=42)
    
    # Retirer la colonne 'delta' avant de séparer
    donnees_concatenees = donnees_concatenees.drop(columns=['delta'])
    
    # Séparer en caractéristiques et labels
    X = donnees_concatenees.iloc[:, :-1]  # Toutes les colonnes sauf 'phase'
    y = donnees_concatenees['phase']  # Colonne 'phase'
    
    # Séparer en ensembles d'apprentissage et de test
    n_train = int(ratio_train * len(X))  # Nombre de points d'apprentissage
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    return X_train, y_train, X_test, y_test

#%% Fonction pour afficher la matrice de confusion et l'accuracy
def evaluer_model(y_test, y_pred, nom_modele):
    """Évalue le modèle en calculant la matrice de confusion, l'accuracy et la balanced accuracy."""
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    print(f"Résultats pour {nom_modele}:")
    print("Matrice de confusion :")
    print(cm)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print("="*50)
    
    return cm, accuracy, balanced_accuracy

#%% Charger les fichiers des sujets
fichiers = ['sujet1.xlsx', 'sujet2.xlsx', 'sujet3.xlsx']  # Mettre les chemins corrects ici
donnees_sujets = charger_donnees_sujets(fichiers)

#%% Mélange aléatoire et préparation des ensembles d'apprentissage et de test
X_train, y_train, X_test, y_test = preparer_ensembles_inter_sujet_random(donnees_sujets, ratio_train=0.7)

#%% Liste des classificateurs à tester
classificateurs = {
    'Quadratic Discriminant Analysis (QDA)': QuadraticDiscriminantAnalysis(),
    'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),
    'Naïve Bayes Gaussien': GaussianNB(),
    'k-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5)
}

#%% Entraîner et évaluer chaque modèle
resultats = {}
for nom_modele, modele in classificateurs.items():
    # Entraîner le modèle
    modele.fit(X_train, y_train)
    
    # Prédire sur le jeu de test
    y_pred = modele.predict(X_test)
    
    # Évaluer et afficher les résultats
    cm, accuracy, balanced_accuracy = evaluer_model(y_test, y_pred, nom_modele)
    
    # Sauvegarder les résultats
    resultats[nom_modele] = {'cm': cm, 'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy}

#%% Comparaison des performances des classificateurs
nom_modeles = list(resultats.keys())
balanced_accuracies = [resultats[modele]['balanced_accuracy'] for modele in nom_modeles]

# Afficher un graphique comparatif des balanced accuracies
plt.figure(figsize=(8, 4))
plt.bar(nom_modeles, balanced_accuracies, color=['turquoise', 'purple', 'pink', 'yellow'])
plt.title('Comparaison des balanced accuracies inter-sujet avec mélange aléatoire')
plt.ylabel('Balanced Accuracy')
plt.xlabel('Classificateur')
plt.xticks(rotation=45, ha="right")
plt.show()
