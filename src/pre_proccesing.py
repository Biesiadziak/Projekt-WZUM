import os
import sys
import time as t
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


import pandas as pd
from ydata_profiling import ProfileReport

# df = pd.read_csv("data/nba_bulk_season_stats_2000_2025.csv")
# profile = ProfileReport(df, title="Profiling Report")
# profile.to_file("nba_stats_profile.html")

MAIN_URL = r"https://www.basketball-reference.com/"
ALPHABET_URL = r"https://www.basketball-reference.com/players/"
ALL_NBA_URL = r"https://www.basketball-reference.com/awards/all_league.html"
ALL_DEFENSIVE_URL= r"https://www.basketball-reference.com/awards/all_defense.html"
MVP_URL = r"https://www.basketball-reference.com/awards/mvp.html"
SEASON_URL = r"https://www.basketball-reference.com/leagues/"

DATA_PATH = r"/home/bartek/Studia/Projekty/WZUM/data"
PLAYER_HTML_PATH = os.path.join(DATA_PATH, "PLAYER_HTML")
AWARD_HTML_PATH = os.path.join(DATA_PATH, "AWARD_HTML")
SEASON_HTML_PATH = os.path.join(DATA_PATH, "SEASON_HTML")

ALPHABET_PATH = os.path.join(DATA_PATH, "Alphabet_Urls.csv")
PLAYER_PATH = os.path.join(DATA_PATH, "Player_Urls.csv")
AWARD_PATH = os.path.join(DATA_PATH, "Award_Urls.csv")
SEASON_PATH = os.path.join(DATA_PATH, "Season_Urls.csv")

PARSER = 'lxml'
ONLY_ACTIVE_PLAYER = True

def filter_out_comment(soup: BeautifulSoup) -> BeautifulSoup:
    content = str(soup).replace('<!--', '')
    content = content.replace('-->', '')
    return BeautifulSoup(content, PARSER)

def request_data(url: str, sleep_time_sec: float = 1.0, with_comment: bool = True) -> BeautifulSoup:
    t.sleep(sleep_time_sec)
    
    if with_comment: 
        return BeautifulSoup(requests.get(url).content, PARSER)
    return filter_out_comment(BeautifulSoup(requests.get(url).content, PARSER))

def season_to_int(cell_value: str):
    if cell_value[-2:] == "00":
        return (int(cell_value[:2]) + 1)*100
    else:
        return int(cell_value[:2] + cell_value[-2:])   


# MVP Voting
content = request_data(url=MVP_URL, sleep_time_sec=4.0, with_comment=False)
table = content.find("table", id="mvp_NBA")
df_table = pd.read_html(str(table))[0]
df_table = df_table.droplevel(0, axis=1)
df_table['Season'] = df_table['Season'].apply(lambda x: season_to_int(x))

votings = []
for td in table.find("tbody").findAll("td", class_="center", attrs={"data-stat":"voting"}):
    votings.append(urljoin(MAIN_URL, td.a['href']))
df_table.insert(loc=len(df_table.columns), column='Voting_Url', value=votings)

df_table = df_table[['Season', 'Voting_Url']]
df_table['Voting_Path'] = df_table['Voting_Url'].apply(lambda cell: os.path.join(AWARD_HTML_PATH, cell.replace("/", "{").replace(":", "}")))

# All NBA
df_table.loc[len(df_table)] = ["All_NBA", ALL_NBA_URL, os.path.join(AWARD_HTML_PATH, ALL_NBA_URL.replace("/", "{").replace(":", "}"))]

# All Defensive
df_table.loc[len(df_table)] = ["All_DEFENSIVE", ALL_DEFENSIVE_URL, os.path.join(AWARD_HTML_PATH, ALL_DEFENSIVE_URL.replace("/", "{").replace(":", "}"))]

df_table.to_csv(AWARD_PATH, index=False, encoding="utf-8-sig")
print("Saved to: ", AWARD_PATH)

df_award = pd.read_csv(AWARD_PATH, encoding="utf-8-sig")
i = 0

for url, path in df_award[['Voting_Url', 'Voting_Path']].values:
    i += 1
    sys.stdout.write(f"\r{i}/{len(df_award)}...")
    
    content = request_data(url=url, sleep_time_sec=4.0, with_comment=False)
    with open(path, "w", encoding='utf-8-sig') as f:
        f.write(str(content))
        f.close()
        
print("\nSaved to: ", AWARD_HTML_PATH, "...")