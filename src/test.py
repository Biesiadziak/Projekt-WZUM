import pandas as pd
import seaborn as sns
import json
import os

data = pd.read_csv("data/advanced_stats.csv")

import matplotlib.pyplot as plt

# Group by Season and AllNBA, then calculate mean for each group
grouped = data.groupby(['AllNBA']).mean(numeric_only=True).reset_index()

# Display the grouped data
print(grouped)

# TS%,WS/48,PER,PTS,TRB,AST,STL,BLK,AllNBA,RookieNBA,Rookie

# Plot average stats per year per AllNBA
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='Season', y='WS/48', hue='AllNBA')
plt.title('Average Points per Season by AllNBA Selection')
plt.ylabel('Average Points')
plt.show()

# Load ground truth player names
with open('ground_truth.json', 'r') as f:
    ground_truth = json.load(f)
# Aggregate all player names from all lists in the JSON
ground_truth_players = set()
for player_list in ground_truth.values():
    ground_truth_players.update(player_list)

# Check each file in results/ folder
results_folder = 'results'
missing_players = {}

for filename in os.listdir(results_folder):
    if filename.endswith('.json'):
        with open(os.path.join(results_folder, filename), 'r') as f:
            data = json.load(f)
        nba_players = set(data.get('NBA', []))
        rookie_players = set(data.get('Rookie', []))
        file_players = nba_players.union(rookie_players)
        not_found = ground_truth_players - file_players
        if not_found:
            missing_players[filename] = list(not_found)

if missing_players:
    for file, players in missing_players.items():
        print(f"Missing players in {file}: {players}")
else:
    print("All players from ground_truth.json are present in every results file.")