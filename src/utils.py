from sklearn.utils import resample
from sklearn.metrics import classification_report
import pandas as pd
import json

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def test_train_split (X, y):
    X_train = X[X["Season"] < 2023].copy()
    X_test = X[X["Season"] == 2023].copy()

    y_train = y[X["Season"] < 2023].copy()
    y_test = y[X["Season"] == 2023].copy()

    X_train.drop(columns=["Season", "Player"], inplace=True)
    X_test.drop(columns=["Season", "Player"], inplace=True)

    return X_train, X_test, y_train, y_test

# --- Downsampling Function ---
def downsample(X_data, y_data, n):
    X_class_0 = X_data[y_data == 0]
    Y_class_0 = y_data[y_data == 0]
    X_non_class_0 = X_data[y_data != 0]
    Y_non_class_0 = y_data[y_data != 0]
    X_down, Y_down = resample(
        X_class_0, Y_class_0,
        n_samples=int(len(X_non_class_0) * n),
        random_state=42
    )
    X_balanced = pd.concat([X_non_class_0, X_down])
    y_balanced = pd.concat([Y_non_class_0, Y_down])
    return X_balanced, y_balanced

def train(X_train, y_train, X_test, y_test, X, label, class_):
    # --- Model Definitions ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    }
    # --- Training and Evaluation Loop ---
    for name, model in models.items():
        print(f"\n{'='*20} {name} {'='*20}")

        # Train main model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=[str(i) for i in range(y_pred_proba.shape[1])])

        # Add player names back
        X_test_players = X[X["Season"] == 2023]["Player"].values
        y_pred_proba_df.index = X_test_players

        # Select teams
        first_team = y_pred_proba_df.sort_values("1", ascending=False).head(30)

        nba_teams = {
            f"{label}": first_team.index.tolist(),
        }

        with open(f"results/all_nba_teams_2023_{name.replace(' ', '_').lower()}.json", "w") as f:
            json.dump(nba_teams, f, indent=2)

        # --- Save filtered output CSV with all statistics for each season ---
        all_seasons_top30 = []
        for season in sorted(X["Season"].unique()):
            output_df = X[X["Season"] == season].copy()
            if output_df.empty:
                continue
            X_season = output_df.drop(columns=["Season", "Player"])
            pred_proba = model.predict_proba(X_season)[:, 1]
            output_df[f"{label}_pred_proba"] = pred_proba
            output_df["Class"] = class_.loc[X["Season"] == season].values
            top_30_season = output_df.sort_values(f"{label}_pred_proba", ascending=False).head(30)
            all_seasons_top30.append(top_30_season)

        combined_top30 = pd.concat(all_seasons_top30, ignore_index=True)
        combined_top30.to_csv(f"data_filtered/{label}_{name.replace(' ', '_').lower()}.csv", index=False)

        print(f"\n{label} Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))