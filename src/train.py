# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from utils import test_train_split, downsample
from utils import train

# --- Data Loading ---
data = pd.read_csv("data/NBA_stats.csv")
data_rookie = pd.read_csv("data/rookies_stats.csv")

data.fillna(0, inplace=True)
data_rookie.fillna(0, inplace=True)

X = data.drop(columns=["AllNBA", "Class"])
y = data["AllNBA"]
class_ = data["Class"]

X_rookie = data_rookie.drop(columns=["RookieNBA", "ClassRookie"])
y_rookie = data_rookie["RookieNBA"]
class_rookie = data_rookie["ClassRookie"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = test_train_split(X, y)
X_train_rookie, X_test_rookie, y_train_rookie, y_test_rookie = test_train_split(X_rookie, y_rookie)

# Uncomment to use downsampling:
# X_train, y_train = downsample(X_train, y_train, 3)
# X_train_rookie, y_train_rookie = downsample(X_train_rookie, y_train_rookie, 2)

# --- Label Distribution ---
print("\n" + "="*40)
print("Label percentages in AllNBA:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nLabel percentages in RookieNBA:")
print(y_train_rookie.value_counts(normalize=True).mul(100).round(2))
print("="*40 + "\n")

train(X_train, y_train, X_test, y_test, X, label="AllNBA", class_=class_)
train(X_train_rookie, y_train_rookie, X_test_rookie, y_test_rookie, X_rookie, label="RookieNBA", class_=class_rookie)