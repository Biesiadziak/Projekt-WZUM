import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('data/all.csv')
print(df_train.columns)
print(df_train['Class'].describe())
sns.distplot(df_train['Class'])
plt.show()

print("Skewness: %f" % df_train['Class'].skew())
print("Kurtosis: %f" % df_train['Class'].kurt())

df_train.drop(columns=["Player", "Awards_adv", "Awards_pg", "Team_adv", "Team_pg", "Pos_adv", "Pos_pg"], inplace=True)
# var = 'TS%'
# data = pd.concat([df_train['Class'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="Class", data=data)
# fig.axis(ymin=0, ymax=4)
# plt.show()

corrmat = df_train.corr()
corr_with_class = corrmat['Class'].drop('Class').abs().sort_values(ascending=False)
print("Variables most correlated with 'Class':")
print(corr_with_class)

top20_vars = corr_with_class.index[20:30]
print("Top 30 most correlated variables:")
print(top20_vars)

corr_top20 = df_train[top20_vars].corr()
print("Correlation matrix of top 30 variables:")
print(corr_top20)

plt.figure(figsize=(14, 12))
sns.heatmap(corr_top20, annot=True, cmap='coolwarm', square=True)
plt.title("Correlation between Top 30 Most Correlated Variables")
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, annot=True, vmax=.9, square=True)
plt.show()

# for name in df_train.columns:
#     var = name.format(str)
#     data = pd.concat([df_train['Class'], df_train[var]], axis=1)
#     data.plot.scatter(x=var, y='Class', ylim=(-0.1, 3.1))
#     plt.show()