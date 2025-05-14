import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/nba_bulk_season_stats_2000_2025.csv")

models = {
    "logreg": (LogisticRegression(max_iter=1000), {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs"]
    }),
    "rf": (RandomForestClassifier(), {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }),
    "svc": (SVC(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    })
}



labels = [""]
df.drop()