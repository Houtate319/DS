
<img src="https://www.encgs.ac.ma/wp-content/uploads/2024/06/logo.png"
     alt="Logo ENCG Settat"
     style="height:100px; margin-right:100px; float:left; border-radius:10px;">

---

# 📘 GRAND GUIDE : ANATOMIE D'UN PROJET DE RÉGRESSION SUR LE DIABETES DATASET

Ce document présente étape par étape le cycle de vie complet d’un projet de Machine Learning de régression appliqué au dataset Diabetes de Scikit-Learn. Il a pour objectif de passer d’une simple exécution de modèle à une compréhension approfondie de l’ensemble de la chaîne : préparation des données, choix du modèle, réglages et interprétation des métriques de performance.

---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Dans le domaine du diabète, les médecins veulent anticiper la **progression de la maladie** à partir de mesures cliniques et biologiques simples, afin de mieux ajuster les traitements et prioriser le suivi des patients à risque.

*   **Objectif :** Construire un modèle de **régression** qui prédit un score continu de progression du diabète (plus le score est élevé, plus la progression est importante).
*   **Enjeu clinique :** Une bonne estimation de cette progression aide à optimiser les ressources médicales, à éviter les sous-traitements pour les profils sévères, et à mieux planifier le suivi des patients.

### Les Données (L'Input)
Nous utilisons le *Diabetes Dataset* intégré à Scikit-Learn via `load_diabetes(as_frame=True)`.

*   **X (Features) :** 10 variables continues normalisées : âge, sexe, indice de masse corporelle (`bmi`), pression artérielle (`bp`), et six mesures sanguines (`s1` à `s6`).
*   **y (Target) :** Variable continue `target` représentant un score de progression de la maladie du diabète.

---

## 2. Le Code Python (Laboratoire)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# --- PHASE 1 : ACQUISITION & SIMULATION ---
data = load_diabetes(as_frame=True)
df = data.frame.copy()
df.rename(columns={'target': 'target'}, inplace=True)

# Simulation de la réalité (Données sales)
np.random.seed(42)
features_columns = [c for c in df.columns if c != "target"]
df_dirty = df.copy()
for col in features_columns:
    df_dirty.loc[df_dirty.sample(frac=0.05, random_state=42).index, col] = np.nan

# --- PHASE 2 : DATA WRANGLING (NETTOYAGE) ---
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

# --- PHASE 3 : ANALYSE EXPLORATOIRE (EDA) ---
print("Statistiques descriptives (toutes les features) :")
print(X_clean.describe())

# --- PHASE 4 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

# --- PHASE 5 : INTELLIGENCE ARTIFICIELLE (RANDOM FOREST REGRESSOR) ---
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- PHASE 6 : AUDIT DE PERFORMANCE ---
y_pred = model.predict(X_test)

from math import sqrt
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f">>> MSE  : {mse:.2f}")
print(f">>> RMSE : {rmse:.2f}")
print(f">>> MAE  : {mae:.2f}")
print(f">>> R²   : {r2:.3f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', label="Idéal"
)
plt.xlabel("Valeurs réelles (y_test)")
plt.ylabel("Prédictions (y_pred)")
plt.title("Random Forest - Prédictions vs Réalité (Diabetes)")
plt.legend()
plt.show()
```

---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### Le Problème Mathématique du "Vide"
Les algorithmes de régression reposent sur des opérations d'algèbre linéaire (produits scalaires, inversions, agrégations) qui ne savent pas gérer la valeur `NaN` (Not a Number).  
Une seule valeur manquante dans une matrice de features peut bloquer l'entraînement ou produire des résultats numériquement instables.

### La Mécanique de l'Imputation
Nous utilisons `SimpleImputer(strategy='mean')` pour reconstruire un tableau de features complet.

1.  **L'Apprentissage (`fit`) :**  
    L'imputer scanne chaque colonne numérique (par exemple `bmi`) sur tous les patients disponibles et calcule la moyenne de cette variable. Il stocke une valeur moyenne par feature dans sa mémoire interne.

2.  **La Transformation (`transform`) :**  
    Lors de la transformation, dès qu'un `NaN` est détecté dans une colonne, il est remplacé par la moyenne pré-apprise pour cette colonne. Le résultat est `X_imputed`, qui est ensuite retransformé en DataFrame `X_clean` avec les mêmes noms de colonnes qu'à l'origine.

### 💡 Le Coin de l'Expert (Data Leakage)
*Attention :* Dans ce script pédagogique, nous avons imputé les valeurs manquantes *avant* de séparer (Train/Test).  

*   *Pourquoi ?* En calculant les moyennes sur l'intégralité de `X`, on utilise indirectement des informations provenant des patients du futur Test Set, ce qui crée une légère **fuite de données** (Data Leakage).
*   *La bonne pratique absolue :*  
    *   Effectuer d'abord le `train_test_split` pour obtenir `X_train` et `X_test`.  
    *   Apprendre l'imputer sur `X_train` uniquement (`imputer.fit(X_train)`), puis appliquer `imputer.transform` sur `X_train` et `X_test` sans recalculer les moyennes sur le test.

---

## 4. Analyse Approfondie : Exploration (EDA)

C'est l'étape de "Profilage" du problème de régression : comprendre les distributions, les échelles et les relations entre les variables explicatives et la cible continue.

### Décrypter `.describe()`
L'appel `X_clean.describe()` affiche pour chaque feature des statistiques comme `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`.

*   **Moyenne (Mean) :** Les moyennes des 10 features sont proches de 0, ce qui est cohérent avec un prétraitement de type centrage-réduction appliqué au dataset Diabetes.
*   **Std (Écart-type) :** Les écarts-types sont tous de l'ordre de 0.045–0.047, montrant que les features sont sur une échelle comparable. Une variable avec un `std` extrêmement faible serait peu informative (quasi constante).

### La Corrélation avec la Cible
Tu visualises la relation entre l'IMC (`bmi`) et la cible `target` via un nuage de points.

*   On observe une tendance globale croissante : plus `bmi` est élevé, plus le score de progression du diabète a tendance à augmenter, ce qui indique une corrélation positive entre surcharge pondérale et aggravation de la maladie.

La matrice de corrélation construite sur `X_clean` et `target` permet de repérer :

*   Des variables fortement corrélées à `target` (candidates intéressantes pour la prédiction).
*   Des groupes de features corrélées entre elles, signalant de la redondance, ce qui est moins critique pour un Random Forest mais important à connaître pour d'autres algorithmes.

---

## 5. Analyse Approfondie : Méthodologie (Split)

### Le Concept : La Garantie de Généralisation
Le but du Machine Learning n'est pas de *mémoriser* les scores passés, mais de *généraliser* correctement à de nouveaux patients.  
Pour simuler ces futurs patients, on isole une partie des données qui ne sera jamais utilisée pendant l'entraînement, mais uniquement lors de l'évaluation finale (jeu de test).

### Les Paramètres sous le capot
`train_test_split(test_size=0.2, random_state=42)`

1.  **Le Ratio 80/20 :**  
    *   80 % des patients (Train) servent au modèle pour apprendre la relation entre les 10 features et le score de progression.
    *   20 % (Test) sont conservés pour mesurer les performances de manière honnête et statistiquement exploitable.

2.  **La Reproductibilité (`random_state`) :**  
    *   Le split est pseudo-aléatoire. Fixer `random_state=42` garantit que chaque exécution du notebook donne la même partition Train/Test.
    *   Cela permet à un autre data scientist, sur une autre machine, d'obtenir exactement la même répartition, ce qui est crucial pour la comparaison et la validation scientifique.

---

## 6. FOCUS THÉORIQUE : L'Algorithme Random Forest 🌲

Pourquoi est-ce l'algorithme "couteau suisse" préféré de nombreux Data Scientists sur des données tabulaires, même en **régression** ?

### A. La Faiblesse de l'Individu (Arbre de Décision de Régression)
Un arbre de décision de régression pose une série de questions pour découper l'espace des patients en régions, puis associe à chaque région une prédiction (souvent la moyenne des cibles des points de cette région).

*   *Problème :* Un arbre unique est **instable**. Il peut sur-apprendre des patients atypiques (outliers) et créer des règles très spécifiques, ce qui entraîne une **forte variance** et des prédictions peu robustes sur de nouvelles données.

### B. La Force du Groupe (Bagging)
Random Forest signifie "Forêt Aléatoire". Il crée de nombreux arbres de décision de régression (ici 200), chacun formé dans un contexte légèrement différent.

1.  **Le Bootstrapping (Diversité des Patients) :**
    *   Chaque arbre est entraîné sur un échantillon bootstrap du jeu d'entraînement (tirage avec remise), donc il ne voit pas exactement les mêmes patients que les autres arbres.
    *   *Conséquence :* Chaque arbre développe une "opinion" différente sur la relation entre les variables cliniques et la progression du diabète.

2.  **Feature Randomness (Diversité des Questions) :**
    *   À chaque split, l'arbre ne considère qu'un sous-ensemble aléatoire de features pour choisir la meilleure coupure, plutôt que l'ensemble complet des 10 colonnes.
    *   *Conséquence :* Cela évite que tous les arbres se focalisent systématiquement sur la même variable dominante et favorise l'exploration de combinaisons de variables moins évidentes.

### C. Le Consensus (Moyenne des Prédictions)
Lorsqu'un nouveau patient arrive :

*   Chaque arbre de la forêt fournit une prédiction réelle (un score de progression du diabète).
*   La prédiction finale du Random Forest Regressor est la **moyenne** de toutes ces prédictions.
*   Les erreurs individuelles des arbres (bruit, sur-apprentissage local) s'annulent en grande partie, ne laissant émerger que la tendance globale (le signal).

---

## 7. Analyse Approfondie : Évaluation (L'Heure de Vérité)

Comment lire les résultats d'un modèle de régression comme un pro ?

### A. Les Métriques de Régression

Sur le jeu de test, tu calcules quatre métriques :

1.  **MSE (Mean Squared Error) :**  
    C'est la moyenne des carrés des erreurs \((y_{test} - y_{pred})^2\). Un MSE bas indique que les prédictions sont en moyenne proches des valeurs réelles, mais la métrique est très sensible aux grosses erreurs.

2.  **RMSE (Root Mean Squared Error) :**  
    La racine carrée du MSE remet l'erreur moyenne au même niveau d'échelle que la cible `target`. Elle permet par exemple de dire : "le modèle se trompe en moyenne d'environ X points de score de progression".

3.  **MAE (Mean Absolute Error) :**  
    C'est la moyenne des valeurs absolues \(|y_{test} - y_{pred}|\). Cette métrique est plus robuste aux outliers et donne une idée intuitive de l'erreur moyenne par patient.

4.  **R² (Coefficient de Détermination) :**  
    Mesure la proportion de variance de `target` expliquée par le modèle (entre 0 et 1). Un R² proche de 1 signifie que le modèle capture bien la structure globale des données, tandis qu'un R² proche de 0 indique un modèle peu informatif.

### B. La Visualisation Prédictions vs Réalité

Tu utilises un scatterplot pour comparer visuellement les prédictions aux valeurs réelles :

*   Chaque point correspond à un patient du jeu de test, avec en abscisse la valeur réelle `y_test` et en ordonnée la prédiction `y_pred`.
*   La droite rouge en pointillés (y = x) représente la prédiction idéale : si tous les points se situaient exactement sur cette diagonale, le modèle serait parfait.

Interprétation :

*   Si la majorité des points est proche de la diagonale, le modèle est bien calibré et les erreurs restent modérées sur l'ensemble des patients.
*   De gros écarts par rapport à la diagonale mettent en évidence des profils de patients pour lesquels le modèle sous-performe, ce qui peut inspirer de futurs raffinements (features supplémentaires, tuning d'hyperparamètres, modèles alternatifs).

### Conclusion du Projet

Ce rapport montre que la Data Science en **régression** ne s'arrête pas à `model.fit()`. C'est une chaîne de décisions logiques où :

*   La compréhension du métier (progression du diabète et impact des erreurs de prédiction sur le suivi des patients) oriente les choix de modèles et de métriques.
*   Les étapes de nettoyage, de profilage (`EDA`), de split et de sélection de l'algorithme (Random Forest pour sa robustesse) sont tout aussi importantes que le code lui-même.
*   L'analyse fine des métriques (MSE, RMSE, MAE, R²) et des graphiques permet de juger si le modèle est réellement utilisable dans un contexte médical ou s'il doit être amélioré.

