# Tools
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Handy for training
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import RFE

# Metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

ground_truth_2023 = {
    "first all-nba team": [
        "Nikola Jokić", "Luka Dončić", "Giannis Antetokounmpo", "Shai Gilgeous-Alexander", "Jayson Tatum"
    ],
    "second all-nba team": [
        "Anthony Davis", "Kevin Durant", "Kawhi Leonard", "Jalen Brunson", "Anthony Edwards"
    ],
    "third all-nba team": [
        "LeBron James", "Stephen Curry", "Domantas Sabonis", "Devin Booker", "Tyrese Haliburton"
    ]
}

team_name_map = {
    1: "first all-nba team",
    2: "second all-nba team",
    3: "third all-nba team"
}

def score_all_nba_predictions(predictions, ground_truth):
    team_levels = {
        "first all-nba team": 1,
        "second all-nba team": 2,
        "third all-nba team": 3
    }

    # Invert the ground truth to map player → true team level
    player_to_true_level = {}
    for team_name, players in ground_truth.items():
        for player in players:
            player_to_true_level[player] = team_levels[team_name]

    total_score = 0

    for predicted_team_name, predicted_players in predictions.items():
        predicted_level = team_levels[predicted_team_name]
        team_score = 0
        correct_team_count = 0

        for player in predicted_players:
            true_level = player_to_true_level.get(player, None)
            if true_level is None:
                points = 0
            else:
                diff = abs(predicted_level - true_level)
                if diff == 0:
                    points = 10
                    correct_team_count += 1
                elif diff == 1:
                    points = 8
                elif diff == 2:
                    points = 6
                else:
                    points = 0

            team_score += points

        # Bonus for correct team assignments
        if correct_team_count == 2:
            team_score += 5
        elif correct_team_count == 3:
            team_score += 10
        elif correct_team_count == 4:
            team_score += 20
        elif correct_team_count == 5:
            team_score += 40

        print(f"{predicted_team_name} → {team_score} points ({correct_team_count} correctly placed in this team)")
        total_score += team_score

    return total_score

def normalize_season(season_str):
    if '-' in season_str:
        return int(season_str[:4])
    return int(season_str)

def encode_five_number(five_str):
    return int(five_str[0])

def get_top_5_predictions(model, X_input, player_names, seasons):
    # Predict class probabilities
    probabilities = model.predict_proba(X_input)
    class_labels = model.classes_
    
    df = pd.DataFrame(probabilities, columns=[f"prob_{cls}" for cls in class_labels])
    df["Player"] = player_names.values
    df["Season"] = seasons.values
    print(df)
    # Build top 5 predictions per class (only for classes 1, 2, 3)
    top_players_by_class = {}
    for class_id in [1, 2, 3]:
        prob_col = f"prob_{class_id}"
        top_df = df.sort_values(by=prob_col, ascending=False).head(5)
        top_players_by_class[class_id] = list(zip(top_df["Player"], top_df["Season"], top_df[prob_col]))

    return top_players_by_class

# Read CSV
X = pd.read_csv("data/nba_bulk_season_stats_total_1946_2024.csv")
Y = pd.read_csv("data/reshaped_nba_players_with_position.csv")

# Delete all empty rows
Y.dropna(how="all", inplace=True) 

# Initialize columns with All NBA stat
X["AllNBA"] = 0
X = X[X["GP"] > 50]

# Convert string data to int data except Player name because dropped later
X["SEASON"] = X["SEASON"].apply(normalize_season)
Y["Season"] = Y["Season"].apply(normalize_season)
Y["Team"] = Y["Team"].apply(encode_five_number)

X["PLAYER_NAME"] = X["PLAYER_NAME"].str.strip()
Y["Player"] = Y["Player"].str.strip()

X = X[X["PLAYER_NAME"].isin(Y["Player"])]

future_season = 2023
X_future = X[X["SEASON"] == future_season].copy()
X = X[X["SEASON"] < future_season]

# Encode if player got the award
for i, row in Y.iterrows():
    player = row["Player"]
    season = row["Season"]
    
    mask = (X["PLAYER_NAME"] == player) & (X["SEASON"] == season)
    X.loc[mask, "AllNBA"] = row["Team"]

# Now, perform the train-test split BEFORE getting player names and seasons
Y = X["AllNBA"]
X.drop(columns=["AllNBA"], inplace=True)
X_future.drop(columns=["AllNBA"], inplace=True)

X_class_0 = X[Y == 0]  # Class 0 players
Y_class_0 = Y[Y == 0]

X_non_class_0 = X[Y != 0]  # Non-Class 0 players (Class 1, 2, 3)
Y_non_class_0 = Y[Y != 0]

# ---------- Downsample Class 0 players to match the size of non-class 0 players ----------

# Resample Class 0 players to match the number of non-class 0 players
X_class_0_downsampled, Y_class_0_downsampled = resample(X_class_0, Y_class_0,
                                                         n_samples=int(len(X_non_class_0)/8),  # Match size of non-class 0
                                                         random_state=42)  # For reproducibility

# ---------- Recombine the datasets ----------

# Concatenate the non-class 0 and downsampled class 0 datasets
X_balanced = pd.concat([X_non_class_0, X_class_0_downsampled])
Y_balanced = pd.concat([Y_non_class_0, Y_class_0_downsampled])

print(X_balanced)
print(Y_balanced)

X_train, X_test, Y_train, Y_test = train_test_split(X_balanced, Y_balanced, test_size=0.2, random_state=42, stratify=Y_balanced)

# Extract player names and seasons after the split to ensure they are aligned with X_test
X_future_player_names = X_future["PLAYER_NAME"]
X_future_season = X_future["SEASON"]


# Drop unnecessary columns
info_to_drop = ["PLAYER_NAME", "NICKNAME", "TEAM_ABBREVIATION", "PLAYER_ID", "TEAM_ID", "SEASON"]

X_train.drop(columns=info_to_drop, inplace=True)
X_test.drop(columns=info_to_drop, inplace=True)
X_future.drop(columns=info_to_drop, inplace=True)

stats_to_drop = ["W_PCT", "FT_PCT", "FG_PCT", "FG3_PCT", "NBA_FANTASY_PTS", "WNBA_FANTASY_PTS", "PLUS_MINUS"]

X_train.drop(columns=stats_to_drop, inplace=True)
X_test.drop(columns=stats_to_drop, inplace=True)
X_future.drop(columns=stats_to_drop, inplace=True)

# Y_train = Y_train[X_train.index]
# Y_test = Y_test[X_test.index]

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 200, 2: 200, 3: 400}),
    "Random Forest": RandomForestClassifier(class_weight={0: 1, 1: 100, 2: 100, 3: 100}),
    "SVM": SVC(class_weight={0: 1, 1: 200, 2: 300, 3: 400})
}

param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],  # Regularization strength
        'solver': ['lbfgs', 'liblinear']  # Optimizer choices
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20],  # Depth of each tree
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],            # Fewer values
        'gamma': ['scale'],          # Only the default
        'kernel': ['rbf', 'linear'] # Kernel coefficient
    }
}

# Iterate through models for grid search
for name, model in models.items():
    print(f"\n{name} - Grid Search Results:")

    # rfe_selector = RFE(estimator=model, n_features_to_select=5)

    # Fit and transform the data
    # X_new = rfe_selector.fit_transform(X_train, Y_train)
    # X_test_new = rfe_selector.transform(X_test)
    # X_future_new = rfe_selector.transform(X_future)
    # Selected features
    # selected_features = rfe_selector.support_

    # Initialize GridSearchCV with the model and its parameter grid
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=1, n_jobs=-1, verbose=3)

    # Fit the model using GridSearchCV
    grid_search.fit(X_train, Y_train)

    # Get the best model after grid search
    best_model = grid_search.best_estimator_

    # Print the best parameters and best score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_}")

    # Make predictions using the best model
    predictions = best_model.predict(X_test)

    # Print accuracy and classification report
    print(f"\n{name} - Best Model Results:")
    print("Accuracy:", accuracy_score(Y_test, predictions))
    print(classification_report(Y_test, predictions))

    # Get the top 5 predictions for each class (0, 1, 2, 3)
    # top_players = get_top_5_predictions(best_model, X_test, Y_test, X_test_player_names, X_test_season)

    # for class_id, players in top_players.items():
    #     print(f"\nTop 5 Players Predicted for Class {class_id}:")
    #     for player, season, prob in players:
    #         print(f"Player: {player}, Season: {season}, Probability: {prob:.4f}")
    top_players = get_top_5_predictions(best_model, X_future, X_future_player_names, X_future_season)
    for class_id, players in top_players.items():
        print(f"\nTop 5 Players Predicted for Class {class_id}:")
        for player, season, prob in players:
            print(f"Player: {player}, Season: {season}, Probability: {prob:.4f}")
    # Generate confusion matrix
    cm = confusion_matrix(Y_test, predictions)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap="Blues")
    
    # Optional: customize plot
    plt.title(f"{name} Confusion Matrix (Best Model)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    # Feature importances from Random Forest
    if name == "Random Forest":
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\nTop 10 Most Important Features (Random Forest):")
        print(feature_importance_df.head(10))

        # Optional: plot
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df["Feature"].head(15), feature_importance_df["Importance"].head(15))
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title("Top 15 Feature Importances - Random Forest")
        plt.tight_layout()
        plt.show()

    predicted_teams_2023 = {
        team_name_map[class_id]: [player for player, _, _ in players]
        for class_id, players in top_players.items()
        if class_id in team_name_map  # skip class 0
    }

    score = score_all_nba_predictions(predicted_teams_2023, ground_truth_2023)
    print(f"\nTotal prediction score for 2023: {score}")