# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from src.utils import test_train_split, downsample, generate_filtered_data, train_model

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import click

@click.command()
@click.argument('path', required=True)

def main(path):
    # --- Data Loading ---
    data = pd.read_csv("data/NBA_stats.csv")
    data_rookie = pd.read_csv("data/rookies_stats.csv")

    data = data[data["G"] >= 64]
    # data_rookie = data_rookie[data_rookie["G"] >= 10]

    data.drop(columns=["G", "GS", "MP", "WS", "2P", "FT"], inplace=True)
    data_rookie.drop(columns=["VORP", "DWS", "PER", "G", "FTA", "PTS"], inplace=True)

    data.fillna(0, inplace=True)
    data_rookie.fillna(0, inplace=True)

    X_all = data.drop(columns=["AllNBA", "Class"])
    y_all = data["AllNBA"]
    class_ = data["Class"]

    X_rookie_all = data_rookie.drop(columns=["RookieNBA", "ClassRookie"])
    y_rookie_all = data_rookie["RookieNBA"]
    class_rookie = data_rookie["ClassRookie"]

    # --- Train-Test Split ---
    X_train_all, X_test_all, y_train_all, y_test_all = test_train_split(X_all, y_all)
    X_train_rookie_all, X_test_rookie_all, y_train_rookie_all, y_test_rookie_all = test_train_split(X_rookie_all, y_rookie_all)

    # Uncomment to use downsampling:
    # X_train, y_train = downsample(X_train, y_train, 3)
    # X_train_rookie, y_train_rookie = downsample(X_train_rookie, y_train_rookie, 2)

    # --- Label Distribution ---
    print("\n" + "="*40)
    print("Label percentages in AllNBA:")
    print(y_train_all.value_counts(normalize=True).mul(100).round(2))

    print("\nLabel percentages in RookieNBA:")
    print(y_train_rookie_all.value_counts(normalize=True).mul(100).round(2))
    print("="*40 + "\n")

    models = {
        #"Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        #"Random Forest": RandomForestClassifier(random_state=42),
        #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
        #"AdaBoost": AdaBoostClassifier(random_state=42),
        #"SVC": SVC(kernel='rbf', probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        #"k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        #"CatBoost": CatBoostClassifier(verbose=0, random_seed=42),
        #"XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42)
    }
    models2 = {
        #"Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
        #"AdaBoost": AdaBoostClassifier(random_state=42),
        #"SVC": SVC(kernel='rbf', probability=True, random_state=42),
        #"Naive Bayes": GaussianNB(),
        #"k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        #"CatBoost": CatBoostClassifier(verbose=0, random_seed=42),
    }

    for model_name, model in models.items():
        print(f"\n{'='*20} {model_name} {'='*20}")

        top30_per_year = generate_filtered_data(model, X_train_all, y_train_all, X_test_all, y_test_all, X_all, label="AllNBA", class_=class_)
        top30_per_year_rookies = generate_filtered_data(model, X_train_rookie_all, y_train_rookie_all, X_test_rookie_all, y_test_rookie_all, X_rookie_all, label="RookieNBA", class_=class_rookie)

        top30_per_year.fillna(0, inplace=True)
        top30_per_year_rookies.fillna(0, inplace=True)

        X = top30_per_year.drop(columns=["Class", "AllNBA_pred_proba"])
        y = top30_per_year["Class"]

        X_rookie = top30_per_year_rookies.drop(columns=["Class", "RookieNBA_pred_proba", "Rookie"])
        y_rookie = top30_per_year_rookies["Class"]

        # --- Train-Test Split ---
        X_train, X_test, y_train, y_test = test_train_split(X, y)
        X_train_rookie, X_test_rookie, y_train_rookie, y_test_rookie = test_train_split(X_rookie, y_rookie)

        for model_name2, model2 in models2.items():
            
            y_pred_proba_df = train_model(model2, X_train, y_train, X_test, X)
            y_pred_proba_df_rookies = train_model(model2, X_train_rookie, y_train_rookie, X_test_rookie, X_rookie)


            first_team = y_pred_proba_df.sort_values("1", ascending=False).head(5)
            remaining_df = y_pred_proba_df.drop(index=first_team.index)

            # Step 2: Second Team - top 5 from class "2", excluding First Team
            second_team = remaining_df.sort_values("2", ascending=False).head(5)
            remaining_df = remaining_df.drop(index=second_team.index)

            # Step 3: Third Team - top 5 from class "3", excluding above
            third_team = remaining_df.sort_values("3", ascending=False).head(5)
            remaining_df = remaining_df.drop(index=third_team.index)

            first_rookie_team = y_pred_proba_df_rookies.sort_values("1", ascending=False).head(5)
            remaining_df = y_pred_proba_df_rookies.drop(index=first_rookie_team.index)

            second_rookie_team = remaining_df.sort_values("2", ascending=False).head(5)
            remaining_df = remaining_df.drop(index=second_rookie_team.index)
            # Build final JSON object
            nba_teams = {
                "first all-nba team": first_team.index.tolist(),
                "second all-nba team": second_team.index.tolist(),
                "third all-nba team": third_team.index.tolist(),
                "first rookie all-nba team": first_rookie_team.index.tolist(),
                "second rookie all-nba team": second_rookie_team.index.tolist()
            }

            # with open(f"results/all_nba_teams_2023_{model.__class__.__name__.replace(' ', '_').lower()}_____{model_name2.replace(' ', '_').lower()}.json", "w") as f:
                    # json.dump(nba_teams, f, indent=2)
            with open(path, "w") as f:
                json.dump(nba_teams, f, indent=2)

if __name__ == "__main__":
    main()