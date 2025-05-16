import pandas as pd

df3 = pd.DataFrame()
for i in range(1989, 2023 + 2):
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


df3.to_csv("rookies.csv", index=False)