---

# 📘 GRAND GUIDE : ANATOMIE D'UN PROJET DATA SCIENCE

Ce document décortique chaque étape du cycle de vie d'un projet de Machine Learning. Il est conçu pour passer du niveau "débutant qui copie du code" au niveau "ingénieur qui comprend les mécanismes internes".

---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Dans le domaine médical, la fatigue des radiologues ou la complexité des images peuvent mener à des erreurs de diagnostic.
*   **Objectif :** Créer un "Assistant IA" pour le second avis médical.
*   **L'Enjeu critique :** La matrice des coûts d'erreur est asymétrique.
    *   Dire à un patient sain qu'il est malade (Faux Positif) génère du stress et des coûts de biopsie.
    *   Dire à un patient malade qu'il est sain (Faux Négatif) peut entraîner la mort par retard de traitement. **L'IA doit donc prioriser la sensibilité (Recall).**

### Les Données (L'Input)
Nous utilisons le *Breast Cancer Wisconsin Dataset*.
*   **X (Features) :** 30 colonnes. Ce ne sont pas des pixels bruts, mais des caractéristiques mathématiques extraites d'images de cellules (Rayon moyen, Écart-type de la texture, "Pire" concavité, etc.).
*   **y (Target) :** Binaire. `0` = Malin, `1` = Bénin.

---

## 2. Le Code Python (Laboratoire)

Ce script est votre paillasse de laboratoire. Il contient toutes les manipulations nécessaires.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# --- PHASE 1 : ACQUISITION & SIMULATION ---
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Simulation de la réalité (Données sales)
np.random.seed(42)
df_dirty = df.copy()
# On corrompt 5% des données avec des NaN
for col in df.columns[:-1]:
    df_dirty.loc[df_dirty.sample(frac=0.05).index, col] = np.nan

# --- PHASE 2 : DATA WRANGLING (NETTOYAGE) ---
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

# Stratégie d'imputation
imputer = SimpleImputer(strategy='mean')
# fit = apprend la moyenne, transform = bouche les trous
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

# --- PHASE 3 : ANALYSE EXPLORATOIRE (EDA) ---
print("--- Statistiques Descriptives ---")
print(X_clean.iloc[:, :5].describe())

# --- PHASE 4 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

# --- PHASE 5 : INTELLIGENCE ARTIFICIELLE (RANDOM FOREST) ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- PHASE 6 : AUDIT DE PERFORMANCE ---
y_pred = model.predict(X_test)

print(f"\n--- Accuracy Globale : {accuracy_score(y_test, y_pred)*100:.2f}% ---")
print("\n--- Rapport Détaillé ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Visualisation des erreurs
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion : Réalité vs IA')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.show()
```

---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### Le Problème Mathématique du "Vide"
Les algorithmes d'algèbre linéaire (qui calculent des distances entre points) ne peuvent pas gérer la valeur `NaN` (Not a Number). Une seule valeur manquante peut faire planter tout le calcul matriciel.

### La Mécanique de l'Imputation
Nous utilisons `SimpleImputer(strategy='mean')`.
1.  **L'Apprentissage (`fit`) :** L'imputer scanne la colonne "Rayon" de tous les patients disponibles. Il calcule $\mu$ (la moyenne), disons 14.12mm. Il stocke cette valeur en mémoire.
2.  **La Transformation (`transform`) :** Il repasse sur les données. S'il voit un trou, il injecte 14.12mm.

### 💡 Le Coin de l'Expert (Data Leakage)
*Attention :* Dans ce script pédagogique, nous avons nettoyé *avant* de séparer (Train/Test). Dans un système industriel ultra-rigoureux, c'est une erreur subtile appelée **Data Leakage** (Fuite de données).
*   *Pourquoi ?* En calculant la moyenne sur tout le monde, la moyenne inclut des infos du futur Test Set.
*   *La bonne pratique absolue :* Séparer d'abord, calculer la moyenne sur le Train, et utiliser cette moyenne "Train" pour boucher les trous du Test.

---

## 4. Analyse Approfondie : Exploration (EDA)

C'est l'étape de "Profilage".

### Décrypter `.describe()`
*   **Mean (Moyenne) vs 50% (Médiane) :** Comparez ces deux lignes. Si la Moyenne est beaucoup plus haute que la Médiane, cela indique une **distribution asymétrique** (skewed) tirée vers le haut par des valeurs extrêmes.
*   **Std (Écart-type) :** Mesure la "largeur" de la cloche de distribution. Une variable avec un std proche de 0 est inutile (c'est une constante).

### La Multicollinéarité (Le problème de la redondance)
En regardant une Heatmap, on verrait que `Radius` (Rayon), `Perimeter` (Périmètre) et `Area` (Aire) sont corrélés à >99%.
*   *Géométriquement :* C'est logique ($P = 2\pi R$).
*   *Impact ML :* Pour un Random Forest, ce n'est pas grave. Mais pour une Régression Logistique, cela rendrait le modèle instable car il ne saurait pas à quelle variable attribuer le "poids" de la décision.

---

## 5. Analyse Approfondie : Méthodologie (Split)

### Le Concept : La Garantie de Généralisation
Le but du Machine Learning n'est pas de *mémoriser* le passé, mais de *généraliser* vers le futur.

### Les Paramètres sous le capot
`train_test_split(test_size=0.2, random_state=42)`
1.  **Le Ratio 80/20 (Le principe de Pareto) :** On garde la majorité des données pour que le modèle puisse capturer la complexité des motifs (Train). On en garde juste assez (Test) pour que la note finale soit statistiquement significative.
2.  **La Reproductibilité (`random_state`) :** En informatique, le "vrai" hasard n'existe pas. C'est du pseudo-aléatoire. Fixer la graine à 42 assure que si vous envoyez votre code à un collègue au Japon, il obtiendra *exactement* les mêmes patients dans son jeu de test. C'est crucial pour la validation scientifique.

---

## 6. FOCUS THÉORIQUE : L'Algorithme Random Forest 🌲

Pourquoi est-ce l'algorithme "couteau suisse" préféré des Data Scientists ?

### A. La Faiblesse de l'Individu (Arbre de Décision)
Un Arbre de Décision unique pose des questions en cascade.
*   *Problème :* Il est **obsessif**. Si, dans vos données d'entraînement, il y a une aberration (un patient sain avec un rayon énorme), l'arbre va créer une règle spécifique pour lui. Il apprend le bruit. On dit qu'il a une **haute variance**.

### B. La Force du Groupe (Bagging)
Random Forest signifie "Forêt Aléatoire". Il crée 100 arbres (ou plus). Pour qu'ils ne soient pas tous identiques, on introduit du chaos contrôlé à deux niveaux :

1.  **Le Bootstrapping (Diversité des Éleves) :**
    *   Chaque arbre ne voit pas tout le monde. L'Arbre #1 s'entraîne sur les patients A, B, C. L'Arbre #2 sur A, C, D.
    *   *Conséquence :* Chaque arbre développe une "opinion" basée sur une expérience différente.

2.  **Feature Randomness (Diversité des Questions) :**
    *   C'est la magie du Random Forest. À chaque fois qu'un arbre veut poser une question pour séparer les malades des sains, il n'a accès qu'à un sous-ensemble aléatoire de colonnes (ex: $\sqrt{nb\_colonnes}$).
    *   *Conséquence :* Cela force les arbres à regarder des variables moins évidentes (comme la texture ou la symétrie) au lieu de se focaliser uniquement sur le rayon.

### C. Le Consensus (Vote)
Lorsqu'un nouveau patient arrive :
*   Les 100 arbres font leur diagnostic individuellement.
*   On fait un vote à la majorité.
*   Les erreurs individuelles des arbres (bruit) s'annulent mathématiquement, ne laissant que la tendance lourde (le signal).

---

## 7. Analyse Approfondie : Évaluation (L'Heure de Vérité)

Comment lire les résultats comme un pro ?

### A. La Matrice de Confusion (Quadrants)
*   **Vrais Positifs (TP) :** *Prédit Cancer | Réel Cancer.* (Succès).
*   **Vrais Négatifs (TN) :** *Prédit Sain | Réel Sain.* (Succès).
*   **Faux Positifs (FP - Erreur de Type I) :** *Prédit Cancer | Réel Sain.*
    *   *Impact :* Stress psychologique, coût.
*   **Faux Négatifs (FN - Erreur de Type II) :** *Prédit Sain | Réel Cancer.*
    *   *Impact :* **Danger de mort.** C'est la métrique à surveiller absolument ici.

### B. Les Métriques Avancées
L'Accuracy (Précision globale) est dangereuse si les classes sont déséquilibrées (ex: 99% de sains).
On regarde donc :

1.  **La Précision (Precision) :** "Qualité de l'alarme".
    $$TP / (TP + FP)$$
    *   Si elle est basse, l'IA crie "Au loup !" trop souvent pour rien.

2.  **Le Rappel (Recall / Sensibilité) :** "Puissance du filet".
    $$TP / (TP + FN)$$
    *   Si elle est basse (ex: 0.60), l'IA laisse passer 40% des cancers. **Inacceptable en médecine.**
    *   *Objectif pro :* Maximiser le Recall, quitte à accepter un peu plus de Faux Positifs.

3.  **F1-Score :** La moyenne harmonique des deux précédents. C'est la note unique la plus honnête pour comparer deux modèles.

### Conclusion du Projet
Ce rapport montre que la Data Science ne s'arrête pas à `model.fit()`. C'est une chaîne de décisions logiques où la compréhension du métier (médecine) dicte le choix des algorithmes (Random Forest pour la robustesse) et des métriques (Recall pour la sécurité).
