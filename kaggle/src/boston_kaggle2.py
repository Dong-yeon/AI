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
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures # 스케일링 및 다항 특성 생성
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from src.boston_kaggle import num_split

# 재현성을 위해 난수 시드를 설정합니다.
np.random.seed(42)

# data MinMaxScaler 및 PolynomialFeatures 적용 함수 정의
def load_extended_boston():
    try:
        boston_house_data = fetch_openml(name="boston", version=1, as_frame=True, parser='liac-arff')
    except Exception as e:
        print(f"데이터셋 로딩 중 오류 발생: {e}")
        print("scikit-learn 1.2 이상 버전에서는 'boston' 데이터셋이 윤리적 문제로 제거되었을 수 있습니다.")
        print("대안 데이터셋(예: 캘리포니아 주택 가격) 사용을 고려해 보세요.")
        exit()

    X = boston_house_data.data.values
    y = boston_house_data.target.values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    return X_poly, y

X_extended, y_target = load_extended_boston()

"""
if X_extended is not None:
    print("--- Extended Features (X_extended) ---")
    # X_extended는 매우 클 수 있으므로 일부만 출력하거나 shape만 확인하는 것이 좋습니다.
    # print(X_extended) # 전체 출력 대신 일부만 출력
    print(X_extended[:2, :15]) # 예시: 첫 2개 샘플의 처음 15개 특성만 출력
    print("\n--- Target (y_target) ---")
    # print(y_target) # 전체 출력
    print(y_target[:10]) # 예시: 첫 10개 타겟 값만 출력
    print("\n--- Shape of Extended Features (X_extended.shape) ---")
    print(X_extended.shape)

    # PolynomialFeatures로 생성된 특성 이름 확인 (선택 사항)
    # 원본 특성 이름 가져오기
    try:
        original_feature_names = fetch_openml(name="boston", version=1, as_frame=True, parser='liac-arff').feature_names
        # PolynomialFeatures 객체를 다시 만들어 get_feature_names_out 호출
        poly_temp = PolynomialFeatures(degree=2, include_bias=False)
        poly_temp.fit(MinMaxScaler().fit_transform(fetch_openml(name="boston", version=1, as_frame=True, parser='liac-arff').data.values)) # fit 과정 필요
        extended_feature_names = poly_temp.get_feature_names_out(original_feature_names)
        print("\n--- Number of Extended Features ---")
        print(len(extended_feature_names))
        print("\n--- Example Extended Feature Names ---")
        print(extended_feature_names[:10]) # 처음 10개 예시
        print(extended_feature_names[-10:]) # 마지막 10개 예시
    except Exception as e:
        print(f"\n특성 이름 생성 중 오류: {e}")
"""

"""
if X_extended is not None:
    num_split = 5 # 폴드 개수 설정
    kf = KFold(n_splits=num_split, shuffle=True, random_state=42) # 데이터를 섞어서 폴드를 나눔
    avg_MSE = 0.0 # 평균 MSE를 저장할 변수 초기화

    print("Starting K-Fold Cross Validation with Extended Features...")
    fold_count = 1
    for train_index, test_index in kf.split(X_extended):
        X_train, X_test = X_extended[train_index], X_extended[test_index]
        y_train, y_test = y_target[train_index], y_target[test_index]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        fold_mse = mean_squared_error(y_test, y_pred)
        avg_MSE += fold_mse
        fold_count += 1

    average_mse_result = avg_MSE / num_split
    average_rmse_result = np.sqrt(average_mse_result)

    print(f'Average MSE : {average_mse_result:.4f}')
    print(f'Average RMSE : {average_rmse_result:.4f}')
    print("-" * 30) # 구분선
"""
"""
if X_extended is not None :
    num_split = 5 # 폴드 개수 설정
    kf = KFold(n_splits=num_split, shuffle=True, random_state=42) # 데이터를 섞어서 폴드를 나눔
    avg_MSE = 0.0 # 평균 MSE를 저장할 변수 초기화

    print("Starting K-Fold Cross Validation with Extended Features...")
    fold_count = 1
    for train_index, test_index in kf.split(X_extended):
        X_train, X_test = X_extended[train_index], X_extended[test_index]
        y_train, y_test = y_target[train_index], y_target[test_index]
        ridge_reg = Ridge(alpha=0.1)
        ridge_reg.fit(X_train, y_train)
        y_pred = ridge_reg.predict(X_test)
        fold_mse = mean_squared_error(y_test, y_pred)
        avg_MSE += fold_mse
        fold_count += 1

    average_mse_result = avg_MSE / num_split
    average_rmse_result = np.sqrt(average_mse_result)

    print(f'Average MSE : {average_mse_result:.4f}')
    print(f'Average RMSE : {average_rmse_result:.4f}')
    print("-" * 30) # 구분선
"""

if X_extended is not None :
    num_split = 5 # 폴드 개수 설정
    kf = KFold(n_splits=num_split, shuffle=True, random_state=42) # 데이터를 섞어서 폴드를 나눔
    avg_MSE = 0.0 # 평균 MSE를 저장할 변수 초기화

    print("Starting K-Fold Cross Validation with Extended Features...")
    fold_count = 1
    for train_index, test_index in kf.split(X_extended):
        X_train, X_test = X_extended[train_index], X_extended[test_index]
        y_train, y_test = y_target[train_index], y_target[test_index]
        lasso_reg = Lasso(alpha=0.1)
        lasso_reg.fit(X_train, y_train)
        y_pred = lasso_reg.predict(X_test)
        fold_mse = mean_squared_error(y_test, y_pred)
        avg_MSE += fold_mse
        fold_count += 1

    average_mse_result = avg_MSE / num_split
    average_rmse_result = np.sqrt(average_mse_result)

    print(f'Average MSE : {average_mse_result:.4f}')
    print(f'Average RMSE : {average_rmse_result:.4f}')
    print("-" * 30) # 구분선
