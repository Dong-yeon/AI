# 필요한 라이브러리들을 가져옵니다.
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold # K-폴드 교차 검증
from sklearn.linear_model import LinearRegression # 선형 회귀 모델
from sklearn.metrics import mean_squared_error # 평균 제곱 오차
import matplotlib.pyplot as plt # 시각화 라이브러리
import seaborn as sns # 시각화 라이브러리
from sklearn.model_selection import train_test_split

# 재현성을 위해 난수 시드를 설정합니다.
np.random.seed(42)

# OpenML에서 보스턴 주택 가격 데이터셋을 불러옵니다. (as_frame=True로 설정하여 Pandas DataFrame으로 불러옴)
# 참고: fetch_openml은 네트워크 연결이 필요하며, 데이터셋 버전에 따라 결과가 달라질 수 있습니다.
# 'parser='liac-arff'' 및 'as_frame=True' 인자를 추가하여 경고를 방지하고 DataFrame 형식을 유지합니다.
try:
    boston_house_data = fetch_openml(name="boston", version=1, as_frame=True, parser='liac-arff')
except Exception as e:
    print(f"데이터셋 로딩 중 오류 발생: {e}")
    print("scikit-learn 1.2 이상 버전에서는 'boston' 데이터셋이 윤리적 문제로 제거되었을 수 있습니다.")
    print("대안 데이터셋(예: 캘리포니아 주택 가격) 사용을 고려해 보세요.")
    # 이 예제에서는 오류 발생 시 실행을 멈추도록 exit()를 사용합니다.
    exit()


# 특성 데이터(X)와 타겟 변수(y)를 분리합니다.
# 데이터가 DataFrame 형식이므로 .values를 사용하여 NumPy 배열로 변환하는 것이 좋습니다.
X = boston_house_data.data.values
y = boston_house_data.target.values

# print(boston_house_data.DESCR) # 데이터셋 설명을 보려면 주석 해제

# --- K-폴드 교차 검증 및 선형 회귀 모델 평가
num_split = 5 # 폴드 개수 설정
kf = KFold(n_splits=num_split, shuffle=True, random_state=42) # 데이터를 섞어서 폴드를 나눔
avg_MSE = 0.0 # 평균 MSE를 저장할 변수 초기화

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    fold_mse = mean_squared_error(y_test, y_pred)
    avg_MSE += fold_mse

average_mse_result = avg_MSE / num_split
average_rmse_result = np.sqrt(average_mse_result)

print(f'Average MSE : {average_mse_result:.4f}')
print(f'Average RMSE : {average_rmse_result:.4f}')
print("-" * 30) # 구분선

# --- 데이터 분석 및 시각화 추가 ---

# 1. 특성과 타겟 변수를 포함하는 Pandas DataFrame 생성
boston_house_df = pd.DataFrame(boston_house_data.data, columns = boston_house_data.feature_names)
boston_house_df['PRICE'] = y
print("DataFrame Head:")
print(boston_house_df.head())
print("-" * 30)

# regplot 오류 방지를 위해 모든 특성 컬럼을 숫자형으로 변환
# 'PRICE'를 제외한 특성 이름 리스트 가져오기
feature_columns = boston_house_df.columns.difference(['PRICE'])
# 각 특성 컬럼에 대해 pd.to_numeric 적용 (errors='coerce'는 변환 불가능한 값을 Na_N으로 처리)
for col in feature_columns:
    boston_house_df[col] = pd.to_numeric(boston_house_df[col], errors='coerce')

# 데이터 타입 변환 후 정보 확인 (선택 사항)
# print("\nData types after conversion:")
# print(boston_house_df.info())
# print("-" * 30)
# !!! 추가된 부분 끝 !!!


# 2. 변수 간 상관관계 분석 및 히트맵 시각화
corr = boston_house_df.corr()
plt.figure(figsize=(10, 10));
sns.heatmap(corr, vmax=0.8, linewidths=0.01, square=True, annot=True, cmap='YlGnBu');
plt.title('Feature Correlation Heatmap');
# plt.show()

# 3. 각 특성과 주택 가격('PRICE') 간의 관계 시각화 (산점도 및 회귀선)
full_column_list = boston_house_df.columns.to_list()
full_column_list.remove('PRICE')
print(f"Features ({len(full_column_list)}): {full_column_list}")
print("-" * 30) # 구분선

figure, ax_list = plt.subplots(nrows=3, ncols=5, figsize=(20, 15))

for i in range(len(full_column_list)):
    row_idx = i // 5
    col_idx = i % 5
    sns.regplot(data=boston_house_df, x=full_column_list[i], y='PRICE', ax=ax_list[row_idx][col_idx],
                scatter_kws={'s': 10}, line_kws={'color': 'red'})
    ax_list[row_idx][col_idx].set_title("regplot " + full_column_list[i])

plt.tight_layout()

# 마지막 2개의 빈 서브플롯을 숨깁니다
if len(full_column_list) < 15:
    for i in range(len(full_column_list), 15):
        row_idx = i // 5
        col_idx = i % 5
        if row_idx < ax_list.shape[0] and col_idx < ax_list.shape[1]: # ax_list 범위 확인
            ax_list[row_idx][col_idx].axis('off') # 축을 숨김
    plt.show() # 숨긴 후 다시 표시할 필요는 보통 없습니다.

useful_feature_list = corr.query("PRICE > 0.5 or PRICE < -0.5").index.values.tolist()
useful_feature_list.remove('PRICE')
print(useful_feature_list)

X = boston_house_df.loc[:,useful_feature_list].values
y = boston_house_df.iloc[:,-1].values

removed_column_list = list(set(full_column_list) - set(useful_feature_list))
print(removed_column_list)

num_split = 5

kf = KFold(n_splits=num_split)

avg_MSE = 0.0

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 선형회귀(Linear Regression) 모델 선언하기
    lr = LinearRegression()

    # 선형회귀(Linear Regression) 모델 학습하기
    lr.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측을 수행합니다.
    y_pred = lr.predict(X_test)

    # MSE(Mean Squared Error)를 측정합니다.
    avg_MSE = avg_MSE + mean_squared_error(y_test, y_pred)

print('Average MSE :', avg_MSE/num_split)
print('Avergae RMSE :', np.sqrt(avg_MSE/num_split))
