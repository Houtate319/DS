---

<img src="https://img.sanishtech.com/u/51352d3b25c7912538c197953bad13e5.jpg"
     alt="HOUATATE Saïd"
     style="height:200px; margin-right:200px; float:left; border-radius:10px;">

---


# 📊 Analyse de Régression : Prédiction de la Progression du Diabète

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-green.svg)](https://jupyter.org/)

## 📋 Aperçu du Projet

Ce projet implémente un **pipeline complet de régression supervisée** sur le dataset **Diabetes** de Scikit-Learn pour prédire la progression de la maladie diabétique à partir de 10 variables cliniques et biologiques.

**Thématique :** Santé / Diabétologie  
**Algorithme principal :** **Random Forest Regressor**  
**Dataset :** 442 patients × 10 features (âge, IMC, pression artérielle, mesures sanguines)

## 🎯 Objectif Métier

Développer un modèle capable d'estimer la **progression quantitative du diabète** (score continu) pour :
- Prioriser le suivi des patients à risque élevé
- Optimiser l'ajustement des traitements
- Aider à la planification des ressources médicales

## 🛠️ Étapes du Pipeline

```
1. Acquisition → Dataset Diabetes (sklearn)
   ↓
2. Simulation → Données "sales" (5% NaN artificiels)
   ↓
3. Nettoyage → Imputation par moyenne (SimpleImputer)
   ↓
4. EDA → Statistiques + Corrélations + Visualisations
   ↓
5. Split → Train/Test (80/20, random_state=42)
   ↓
6. Modélisation → RandomForestRegressor (200 arbres)
   ↓
7. Évaluation → MSE, RMSE, MAE, R² + Prédictions vs Réalité
```

