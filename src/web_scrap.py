import os
import sys
import time as t
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = r"https://www.basketball-reference.com/leagues/NBA_2025_advanced.html"
MVP_URL = r"https://www.basketball-reference.com/awards/mvp.html"

PARSER = 'lxml'
ONLY_ACTIVE_PLAYER = False

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
# content = request_data(url=MVP_URL, sleep_time_sec=4.0, with_comment=False)
# table = content.find("table", id="mvp_NBA")
# df_table = pd.read_html(str(table))[0]
# df_table = df_table.droplevel(0, axis=1)
# df_table['Season'] = df_table['Season'].apply(lambda x: season_to_int(x))


# df_table.to_csv("test.csv", index=False, encoding="utf-8-sig")
# print("Saved to: ", "test.csv")

# MVP Voting
def get_advanced_stats():
    for i in range(1989, 2026):
        print(i)
        url = f"https://www.basketball-reference.com/leagues/NBA_{i}_advanced.html"
        content = request_data(url=url, sleep_time_sec=4.0, with_comment=False)
        table = content.find("table", id="advanced")
        df_table = pd.read_html(str(table))[0]
        df_table['Season'] = i-1

        df_table.to_csv(f"data_adv/test_{i}.csv", index=False, encoding="utf-8-sig")
        print("Saved to: ", f"data_{i}.csv")

def get_per_game_stats():
    for i in range(1989, 2026):
        print(i)
        url = f"https://www.basketball-reference.com/leagues/NBA_{i}_per_game.html"
        content = request_data(url=url, sleep_time_sec=4.0, with_comment=False)
        table = content.find("table", id="per_game_stats")
        df_table = pd.read_html(str(table))[0]
        df_table['Season'] = i-1
        df_table.to_csv(f"data_per_game/per_game_{i}.csv", index=False, encoding="utf-8-sig")
        print("Saved to: ", f"per_game_{i}.csv")

get_per_game_stats()