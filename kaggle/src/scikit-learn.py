import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression  # LogisticRegression 대신 LinearRegression 사용
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# 맑은 고딕 폰트 설정
plt.rc('font', family='Malgun Gothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

url = "https://raw.githubusercontent.com/solaris33/data-science-for-all/main/lecture_4/weight-height.csv"
df = pd.read_csv(url)

df['Height'] = df['Height'].apply(lambda x: x * 2.54) # cm로 변환
df['Weight'] = df['Weight'].apply(lambda x: x * 0.453592) # kg로 변환

# 이상치 제거
df = df[(df['Height'] > 120) & (df['Height'] < 220)]
df = df[(df['Weight'] > 30) & (df['Weight'] < 150)]

# 성별을 숫자로 변환
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

# 데이터 준비
x = df[['Gender', 'Height', 'BMI']]
y = df['Weight']

# 훈련 데이터와 테스트 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 모델 훈련
# lr = LinearRegression()
# lr.fit(x_train, y_train)

# 예측
# y_pred = lr.predict(x_test)

# 성능 평가
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# 시각화 (실제 vs 예측)
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred)
# plt.xlabel('실제 몸무게 (kg)')
# plt.ylabel('예측 몸무게 (kg)')
# plt.title('실제 vs 예측 몸무게')
# plt.grid(True)
# plt.show()

train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

# LightGBM 모델 훈련
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 모델 훈련
model = lgb.train(params, train_data, 100, valid_sets=[test_data])

# 예측
y_pred = model.predict(x_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('실제 몸무게 (kg)')
plt.ylabel('예측 몸무게 (kg)')
plt.title('실제 vs 예측 몸무게')
plt.grid(True)
plt.show()

# 선형회귀는 입력 변수와 출력 변수 사이가 직선일 경우 잘 맞음.
# 키 / 몸무게 데이터는 직선이 아닐 경우가 있음.
# LightGBM은 비선형 회귀도 가능함. 결정트리 기반.
