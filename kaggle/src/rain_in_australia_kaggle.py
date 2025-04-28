import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(data_path):
    """데이터를 로드하고 기본 정보를 출력하는 함수"""
    df = pd.read_csv(data_path)
    return df

def explore_data(df):
    """데이터 탐색을 위한 함수"""
    print("\n=== Head of the dataset ===\n")
    print(df.head())
    print("\n=== Dataset Info ===\n")
    print(df.info())
    print("\n=== Missing Values ===\n")
    print(df.isnull().sum())

def visualize_target(df):
    """타겟 변수 분포 시각화 함수"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x="RainTomorrow", data=df)
    plt.title("Distribution of Rain Tomorrow")
    plt.show()

def feature_engineering(df):
    """특성 공학 및 상관관계 분석 함수"""
    # 데이터 복사
    temp_df = df.copy()
    
    # 범주형 변수 처리
    categorical_cols = ['RainTomorrow', 'RainToday']
    for col in categorical_cols:
        temp_df[col] = temp_df[col].map({'Yes': 1, 'No': 0})
    
    # 숫자형 변수만 선택
    numeric_cols = temp_df.select_dtypes(include=['float64', 'int64']).columns
    temp_df = temp_df[numeric_cols]
    
    # 결측치 처리
    temp_df = temp_df.fillna(temp_df.mean())
    
    # 상관관계 분석
    try:
        corr = temp_df.corr()
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # RainTomorrow와의 상관관계가 높은 변수들 출력
        rain_corr = corr['RainTomorrow'].sort_values(ascending=False)
        print("\nTop correlations with RainTomorrow:")
        print(rain_corr)
        
    except Exception as e:
        print(f"Error in correlation analysis: {str(e)}")
        print("\nDataFrame info:")
        print(temp_df.info())

# 데이터 전처리
def data_preprocessing(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # print(categorical_cols)
    # print(numeric_cols)

    # 결측치 개수 확인
    # print(df[categorical_cols].isnull().mean().sort_values(ascending=False))
    # print(df[numeric_cols].isnull().mean().sort_values(ascending=False))

    # 결측치 처리
    for col in numeric_cols:
        if df[col].isnull().mean() > 0:
            col_median = df[col].median()
            df[col].fillna(col_median, inplace=True)

    for col in categorical_cols:
        print(col, (df[col].isnull().mean()))
    
    # categorical value column은 최빈값으로 결측치 채우기
    df['WindGustDir'].fillna(df['WindGustDir'].mode()[0], inplace=True)
    df['WindDir9am'].fillna(df['WindDir9am'].mode()[0], inplace=True)
    df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0], inplace=True)
    df['RainToday'].fillna(df['RainToday'].mode()[0], inplace=True)

    # 날짜 column drop
    df.drop(['Date'], axis=1, inplace=True)
    df.dropna(how='any', inplace=True)
    
    # Yes/No 값을 0,1로 변환
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

    # categorical value에 one-hot encoding 적용
    df = pd.get_dummies(df, columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

    # trianing data
    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 결정트리(Decision Tree) 비 예측
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    # 성능 평가
    print(score)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    print("\n\n")
    
    # 랜덤 포레스트
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    # 성능 평가
    print(score)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    return df

def main():
    # 데이터 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), 'dataset', 'weatherAUS.csv')
    
    # 데이터 로드
    df = load_data(data_path)
    
    # 데이터 탐색
    # explore_data(df)
    
    # 데이터 전처리
    df = data_preprocessing(df)
    
    # 데이터 시각화
    # visualize_target(df)
    
    # 상관분석
    # feature_engineering(df)

if __name__ == "__main__":
    main()