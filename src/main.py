# collect_bulk_season_stats.py

import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players
from tqdm import tqdm

def get_season_stats(season, season_type="Regular Season"):
    """Fetch all players' stats for one season."""
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed="Totals"
    )
    df = stats.get_data_frames()[0]
    df["SEASON"] = season
    return df

def main():
    start_year = 1946
    end_year = 2024  # You can adjust this for more/less seasons

    all_seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year)]
    all_data = []

    for season in tqdm(all_seasons, desc="Fetching season data"):
        try:
            df = get_season_stats(season)
            all_data.append(df)
            time.sleep(1.0)  # still polite with API
        except Exception as e:
            print(f"❌ Error for season {season}: {e}")
            continue

    combined_df = pd.concat(all_data, ignore_index=True)

    # Save full dataset
    combined_df.to_csv("nba_bulk_season_stats_total_1946_2024.csv", index=False)
    print("✅ Data saved to nba_bulk_season_stats_total_1946_2024.csv")

if __name__ == "__main__":
    main()
