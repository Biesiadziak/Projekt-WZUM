import os
import sys
import time as t
import numpy as np
import pandas as pd

CURRENT_YEAR = 2023

def normalize_season(season_str):
    if '-' in season_str:
        return int(season_str[:4])
    return int(season_str)

def encode_five_number(five_str):
    return int(five_str[0])

def nba_com():
    x_file = f"data/nba_bulk_season_stats_1996_2024.csv"
    y_file = f"data/all_nba_results.csv"

    df_table = pd.read_csv(x_file)
    temp = pd.read_csv(y_file)

    temp.dropna(how="all", inplace=True) 

    df_table["SEASON"] = df_table["SEASON"].apply(normalize_season)
    temp["Season"] = temp["Season"].apply(normalize_season)
    temp["Team"] = temp["Team"].apply(encode_five_number)

    df_table["PLAYER_NAME"] = df_table["PLAYER_NAME"].str.strip()
    temp["Player"] = temp["Player"].str.strip()

    df_table = df_table[df_table["GP"] > 10]

    df_table["AllNBA"] = 0

    for i, row in temp.iterrows():
        player = row["Player"]
        season = row["Season"]
        
        mask = (df_table["PLAYER_NAME"] == player) & (df_table["SEASON"] == season)
        df_table.loc[mask, "AllNBA"] = row["Team"]

    #Split to train and test
    df_test = df_table[df_table['SEASON'] == CURRENT_YEAR]
    df_table = df_table[df_table['SEASON'] < CURRENT_YEAR]

    to_keep = ["SEASON", "PLAYER_NAME", "PTS", "REB", "AST", "STL", "BLK", "AllNBA"]

    df_table = df_table[to_keep]
    df_test = df_test[to_keep]

    df_table.to_csv(f"train_data_test.csv", index=False)


def basketball_ref():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    for i in range(1989, CURRENT_YEAR + 2):
        file = f"data_adv/test_{i}.csv"
        df_table = pd.read_csv(file)
        df1 = pd.concat([df1, df_table], ignore_index=True)

        file = f"data_per_game/per_game_{i}.csv"
        df_table = pd.read_csv(file)
        df2 = pd.concat([df2, df_table], ignore_index=True)

        file = f"data_rookies/rookies_{i}.csv"
        df_table = pd.read_csv(file, header=1)
        df3 = pd.concat([df3, df_table], ignore_index=True)

    # Rename the 'Unnamed: 30' column to 'Season'
    df3.rename(columns={"Unnamed: 30": "Season"}, inplace=True)

    # Strip whitespace from player names (if any) and drop missing values
    df3["Player"] = df3["Player"].str.strip()
    df3["Season"] = df3["Season"].astype(str).str.strip()

    # Keep only Player and Season
    df3 = df3[["Player", "Season"]].dropna()

    df3 = df3[df3["Player"] != "Player"]

    # Optional: remove leading/trailing whitespace
    df3["Player"] = df3["Player"].str.strip()

    # Reset index (cleaner output)
    df3.reset_index(drop=True, inplace=True)

    df1['Player'] = df1['Player'].str.strip()
    df2['Player'] = df2['Player'].str.strip()
    df2['Player'] = df2['Player'].str.strip()

    df1.dropna(subset=["Rk"], inplace=True)
    df2.dropna(subset=["Rk"], inplace=True)
    
    to_keep_adv = ["Player", "Season", "WS/48", "PER"]
    to_keep_per_game = ["Player", "Season", "Team", "PTS", "TRB", "AST", "STL", "BLK"]

    df1 = df1[to_keep_adv]
    df2 = df2[to_keep_per_game]

    merged_df = pd.merge(df1, df2, on=['Player', 'Season'], how='inner', suffixes=('_adv', '_pg'))

    merged_df = merged_df.drop_duplicates(subset=['Player', 'Season'], keep='first')
    merged_df.drop(columns=['Team'], inplace=True)
    merged_df_rookie = merged_df.copy()
    merged_df["AllNBA"] = 0
    merged_df["Class"] = 0
    merged_df["Rookie"] = 0
    merged_df_rookie["RookieNBA"] = 0
    merged_df_rookie["ClassRookie"] = 0
    merged_df_rookie["Rookie"] = 0

    y_file = f"data/all_nba_results.csv"
    y_rookie_file = f"data/all_rookies_players.csv"

    temp2 = pd.read_csv(y_rookie_file)
    
    temp2.dropna(how="all", inplace=True)

    all_rookies = []

    for idx, row in temp2.iterrows():
        season = normalize_season(row['Season'])
        team = row['Tm']

        for col in row.index:
            if "Unnamed" in col and pd.notna(row[col]):
                player = row[col].strip()
                all_rookies.append({'Season': season, 'Player': player, 'Team': team})

    temp2 = pd.DataFrame(all_rookies)

    temp = pd.read_csv(y_file)

    temp.dropna(how="all", inplace=True)
    temp["Season"] = temp["Season"].apply(normalize_season)
    temp["Team"] = temp["Team"].apply(encode_five_number)
    temp["Player"] = temp["Player"].str.strip()
    
    temp2["Team"] = temp2["Team"].apply(encode_five_number)
    temp2["Player"] = temp2["Player"].str.strip()

    df3["Season"] = df3["Season"].apply(normalize_season)
    
    for i, row in temp.iterrows():
        player = row["Player"]
        season = row["Season"]
        
        mask = (merged_df["Player"] == player) & (merged_df["Season"] == season)
        merged_df.loc[mask, "AllNBA"] = 1
        merged_df.loc[mask, "Class"] = row["Team"]

    for i, row in temp2.iterrows():
        player = row["Player"]
        season = row["Season"]
        mask = (merged_df_rookie["Player"] == player) & (merged_df_rookie["Season"] == season)
        merged_df_rookie.loc[mask, "RookieNBA"] = 1
        merged_df_rookie.loc[mask, "ClassRookie"] = row["Team"]

    for i, row in df3.iterrows():
        player = row["Player"].strip()
        season = int(row["Season"])

        player = row["Player"]
        season = row["Season"]
        mask = (merged_df_rookie["Player"] == player) & (merged_df_rookie["Season"] == season)
        merged_df.loc[mask, "Rookie"] = 1
        merged_df_rookie.loc[mask, "Rookie"] = 1

    # Drop all rookies from merged_df (keep only non-rookies)
    merged_df = merged_df[merged_df["Rookie"] == 0]
    merged_df.drop(columns=["Rookie"], inplace=True)
    # Drop all non-rookies from merged_df_rookie (keep only rookies)
    merged_df_rookie = merged_df_rookie[merged_df_rookie["Rookie"] == 1]

    merged_df.to_csv(f"data/NBA_stats.csv", index=False)
    merged_df_rookie.to_csv(f"data/rookies_stats.csv", index=False)
    
    

basketball_ref()