# Tools
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Handy for training
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

data = pd.read_csv("data/advanced_stats.csv")

X = data.drop(columns=["AllNBA", "RookieNBA", "Class", "ClassRookie"])
y = data["Class"]
y_rookie = data["ClassRookie"]

X.fillna(0, inplace=True)

# Train-test split
X_train = X[X["Season"] < 2023].copy()
X_test = X[X["Season"] == 2023].copy()
X_train_rookie = X_train.copy()
X_test_rookie = X_test.copy()

y_train = y[X["Season"] < 2023].copy()
y_test = y[X["Season"] == 2023].copy()
y_train_rookie = y_rookie[X["Season"] < 2023].copy()
y_test_rookie = y_rookie[X["Season"] == 2023].copy()

X_train.drop(columns=["Season", "Player"], inplace=True)
X_test.drop(columns=["Season", "Player"], inplace=True)
X_train_rookie.drop(columns=["Season", "Player"], inplace=True)
X_test_rookie.drop(columns=["Season", "Player"], inplace=True)

# Downsampling Class 0 (majority class)
def downsample(X_data, y_data, n):
    X_class_0 = X_data[y_data == 0]
    Y_class_0 = y_data[y_data == 0]
    X_non_class_0 = X_data[y_data != 0]
    Y_non_class_0 = y_data[y_data != 0]
    
    X_down, Y_down = resample(X_class_0, Y_class_0,
                              n_samples=int(len(X_non_class_0) / n),
                              random_state=42)
    X_balanced = pd.concat([X_non_class_0, X_down])
    y_balanced = pd.concat([Y_non_class_0, Y_down])
    return X_balanced, y_balanced


X_train, y_train = downsample(X_train, y_train, 3)
X_train_rookie, y_train_rookie = downsample(X_train_rookie, y_train_rookie, 2)


label_counts = y_train.value_counts(normalize=True) * 100
print("Label percentages in AllNBA:")
print(label_counts)

rookie_label_counts = y_train_rookie.value_counts(normalize=True) * 100
print("Label percentages in RookieNBA:")
print(rookie_label_counts)

# Class weights
weights = {0: 1, 1: 10}

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

for name, model in models.items():
    print(f"\n{name} - Training and Evaluation:")

    # Train main model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=[str(i) for i in range(y_pred_proba.shape[1])])
    
    # Train rookie model
    rookie_model = model.__class__(**model.get_params())
    rookie_model.fit(X_train_rookie, y_train_rookie)
    y_pred_rookie = rookie_model.predict(X_test_rookie)
    y_pred_proba_rookie = rookie_model.predict_proba(X_test_rookie)
    y_pred_proba_rookie_df = pd.DataFrame(y_pred_proba_rookie, columns=[str(i) for i in range(y_pred_proba_rookie.shape[1])])

    # Add player names back
    X_test_players = X[X["Season"] == 2023]["Player"].values
    y_pred_proba_df.index = X_test_players
    y_pred_proba_rookie_df.index = X_test_players

    # Select teams
    first_team = y_pred_proba_df.sort_values("1", ascending=False).head(5)
    remaining_df = y_pred_proba_df.drop(index=first_team.index)

    # Step 2: Second Team - top 5 from class "2", excluding First Team
    second_team = remaining_df.sort_values("2", ascending=False).head(5)
    remaining_df = remaining_df.drop(index=second_team.index)

    # Step 3: Third Team - top 5 from class "3", excluding above
    third_team = remaining_df.sort_values("3", ascending=False).head(5)
    remaining_df = remaining_df.drop(index=third_team.index)

    first_rookie_team = y_pred_proba_rookie_df.sort_values("1", ascending=False).head(5)
    remaining_df = y_pred_proba_rookie_df.drop(index=first_rookie_team.index)

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

    with open(f"results/all_nba_teams_2023_{name.replace(' ', '_').lower()}.json", "w") as f:
        json.dump(nba_teams, f, indent=2)

    print("\nAll-NBA Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nRookie-NBA Classification Report:")
    print(classification_report(y_test_rookie, y_pred_rookie))