# Tools
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json


# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Handy for training
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.metrics import make_scorer

def custom_score_exclude_class_0(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, labels=[1, 2, 3], average='weighted')

data = pd.read_csv("data/advanced_stats.csv")

X = data.drop(columns=["AllNBA"])
y = data["AllNBA"]

X.fillna(0, inplace=True)

X_train = X[X["Season"] < 2023].copy()
X_test = X[X["Season"] == 2023].copy()
y_train = y[X["Season"] < 2023].copy()
y_test = y[X["Season"] == 2023].copy()
X_train.drop(columns=["Season", "Player"], inplace=True)
X_test.drop(columns=["Season", "Player"], inplace=True)

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(zip(np.unique(y_train), weights))


scorer = make_scorer(custom_score_exclude_class_0)
weights = {0: 1, 1: 10000, 2: 20000, 3: 30000}
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight=weights, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight=weights, random_state=42
    ),
    "SVM": SVC(
        class_weight=weights, probability=True, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    ),
    "LightGBM": LGBMClassifier(random_state=42)
}

param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "gamma": ["scale"],
        "kernel": ["rbf", "linear"]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1]
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "num_leaves": [31, 64],
        "learning_rate": [0.01, 0.1]
    }
}

# Iterate through models for grid search
for name, model in models.items():
    print(f"\n{name} - Grid Search Results:")
    # model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={1: 100, 2: 200, 3: 300, 0: 1})

    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring=scorer, cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    # Get the best model after grid search
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=["0", "1", "2", "3"])

    X_test_players = X[X["Season"] == 2023]["Player"].values
    y_pred_proba_df.index = X_test_players

    # # Ensure column names are strings
    y_pred_proba_df.columns = y_pred_proba_df.columns.astype(str)

    # Step 1: First Team - top 5 from class "1"
    first_team = y_pred_proba_df.sort_values("1", ascending=False).head(5)
    remaining_df = y_pred_proba_df.drop(index=first_team.index)

    # Step 2: Second Team - top 5 from class "2", excluding First Team
    second_team = remaining_df.sort_values("2", ascending=False).head(5)
    remaining_df = remaining_df.drop(index=second_team.index)

    # Step 3: Third Team - top 5 from class "3", excluding above
    third_team = remaining_df.sort_values("3", ascending=False).head(5)

    # Build final JSON object
    nba_teams = {
        "first all-nba team": first_team.index.tolist(),
        "second all-nba team": second_team.index.tolist(),
        "third all-nba team": third_team.index.tolist()
    }
    
    # weights = {
    #     "0": -0.5,
    #     "1": 3,
    #     "2": 2,
    #     "3": 1
    # }

    # # Compute weighted score
    # weighted_scores = (
    #     y_pred_proba_df["0"] * weights["0"] +
    #     y_pred_proba_df["1"] * weights["1"] +
    #     y_pred_proba_df["2"] * weights["2"] +
    #     y_pred_proba_df["3"] * weights["3"]
    # )

    # # Add scores to dataframe
    # y_pred_proba_df["score"] = weighted_scores

    # # Sort by score
    # sorted_df = y_pred_proba_df.sort_values("score", ascending=False)

    # # Select top 15 players overall, then split into 3 teams
    # top_15 = sorted_df.head(15)

    # nba_teams = {
    #     "first all-nba team": top_15.index[:5].tolist(),
    #     "second all-nba team": top_15.index[5:10].tolist(),
    #     "third all-nba team": top_15.index[10:15].tolist()
    # }

    # Save to JSON file
    with open(f"results/all_nba_teams_2023_{name.replace(' ', '_').lower()}.json", "w") as f:
        json.dump(nba_teams, f, indent=2)

    print(classification_report(y_test, y_pred))