# usa_housing_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42  # graine pour rendre les résultats reproductibles

def load_data(path="USA_Housing.csv"):
    """Charge les données et affiche un aperçu."""
    df = pd.read_csv(path)              # lecture du CSV
    print("Aperçu des données :")
    print(df.head())                    # premières lignes
    print("\nInfo :")
    print(df.info())                    # types et valeurs manquantes
    return df

def eda(df):
    """Analyse exploratoire rapide : distribution + corrélations."""
    # Histogramme de la variable cible Price
    plt.figure(figsize=(6,4))
    sns.histplot(df["Price"], kde=True)
    plt.title("Distribution de la variable cible Price")
    plt.tight_layout()
    plt.show()

    # Matrice de corrélation pour les variables numériques
    plt.figure(figsize=(10,8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()

def split_features_target(df):
    """Sépare X (features) et y (cible)."""
    target_col = "Price"                # colonne cible
    X = df.drop(columns=[target_col])   # toutes les colonnes sauf Price
    y = df[target_col]                  # cible
    return X, y

def build_preprocessor(X):
    """Construit le préprocesseur num + cat (imputation + scaling)."""
    # Sélection des colonnes numériques et catégorielles
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Pipeline numérique : imputation médiane + standardisation
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline catégoriel : imputation par la modalité la plus fréquente
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
        # on pourrait ajouter OneHotEncoder ici si besoin
    ])

    # ColumnTransformer pour appliquer le bon pipeline à chaque type de variable
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )
    return preprocessor, num_cols, cat_cols

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Entraîne plusieurs modèles de régression et compare leurs performances."""
    # Dictionnaire des modèles à tester
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    }

    results = []  # liste pour stocker RMSE et R² de chaque modèle

    for name, model in models.items():
        # Pipeline complet : prétraitement + modèle
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)           # entraînement
        y_pred = pipe.predict(X_test)        # prédictions sur le test

        # Calcul des métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({"model": name, "rmse": rmse, "r2": r2})
        print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2:.3f}")

    # Tableau récapitulatif trié par RMSE croissante
    results_df = pd.DataFrame(results).sort_values(by="rmse")
    print("\nTableau comparatif des performances :")
    print(results_df)

    # Récupération du nom du meilleur modèle (plus faible RMSE)
    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    print(f"\nMeilleur modèle: {best_name}")

    # Réentraîner un pipeline complet pour le meilleur modèle
    best_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", best_model)
    ])
    best_pipe.fit(X_train, y_train)
    y_pred_best = best_pipe.predict(X_test)

    # Nuage de points y_test vs y_pred pour juger la qualité des prédictions
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_best, alpha=0.5)
    max_val = max(y_test.max(), y_pred_best.max())
    plt.plot([0, max_val], [0, max_val], "r--")   # diagonale parfaite
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(f"y_test vs y_pred – {best_name}")
    plt.tight_layout()
    plt.show()

    # Importance des variables si le meilleur modèle est basé sur des arbres
    if best_name in ["RandomForest", "GradientBoosting"]:
        # On ajuste le préprocesseur seul pour récupérer les features transformées
        preprocessor.fit(X_train)
        X_train_trans = preprocessor.transform(X_train)

        # Entraînement direct du modèle sur les données transformées
        importances = best_model.fit(X_train_trans, y_train).feature_importances_

        # Récupération des noms de colonnes (num + cat)
        feature_names = (
            preprocessor.transformers_[0][2] +  # noms numériques
            preprocessor.transformers_[1][2]    # noms catégoriels
        )

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Barplot des 10 features les plus importantes
        plt.figure(figsize=(8,5))
        sns.barplot(data=feat_imp.head(10), x="importance", y="feature")
        plt.title(f"Top 10 features – {best_name}")
        plt.tight_layout()
        plt.show()

    return results_df

def main():
    """Fonction principale : charge les données, EDA, split, modèles."""
    df = load_data("USA_Housing.csv")      # chargement du dataset
    eda(df)                                # EDA rapide

    X, y = split_features_target(df)       # séparation X / y

    # Découpage en train/test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor, _, _ = build_preprocessor(X)  # préprocesseur commun

    # Entraînement et évaluation des modèles
    results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

if __name__ == "__main__":
    main()  # lance l’analyse si le script est exécuté directement

