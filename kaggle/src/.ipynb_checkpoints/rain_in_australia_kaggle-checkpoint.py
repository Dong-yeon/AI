#%% [markdown]
# # 호주 강우량 데이터 분석

#%% [라이브러리 불러오기]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(script_dir), 'dataset', 'weatherAUS.csv')

df = pd.read_csv(data_path)

print("\n=== Head of the dataset ===\n")
print(df.head())
print(df.info())
print(df.isnull().sum())

sns.countplot(x="RainTomorrow", data=df)
plt.show()
