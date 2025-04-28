import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 개수 확인
def count_pass_fail(input):
    nb_pass = 0
    nb_pass_half = 0
    nb_defective = 0
    for i in range(len(input)):
        if input[i,5] == 'no':
            nb_defective += 1
        if input[i,5] == 'yes' and input[i,6] == 'yes':
            nb_pass += 1
        if input[i,5] == 'yes' and input[i,6] == 'no':
            nb_pass_half += 1

    # print ("nb_pass:", nb_pass)
    # print ("nb_pass_half:", nb_pass_half)
    # print ("nb_defective:", nb_defective)
    # print ("total:", nb_pass + nb_pass_half + nb_defective)
    return nb_pass, nb_pass_half, nb_defective

# 데이터 전처리
def tool_condition(input):
    for i in range(len(input)):
        if input[i,4] == 'unworn':
            input[i,4] = 0
        else:
            input[i,4] = 1
    return input

# train 데이터의 육안검사 결과를 0,1,2로 변환
def item_inspection(input):
    for i in range(len(input)):
        if input[i,5] == 'no':
            input[i,6] = 2
        elif input[i,5] == 'yes' and input[i,6] == 'no':
            input[i,6] = 1
        elif input[i,5] == 'yes' and input[i,6] == 'yes':
            input[i,6] = 0
    return input

# 원본 데이터의 공정을 0,1,2,3,4,5,6,7,8,9로 변환
def machining_process(input):
    for i in range(len(input)):
        if input[i,47] == 'Prep':
            input[i,47] = 0
        elif input[i,47] == 'Layer 1 Up':
            input[i,47] = 1
        elif input[i,47] == 'Layer 1 Down':
            input[i,47] = 2
        elif input[i,47] == 'Layer 2 Up':
            input[i,47] = 3
        elif input[i,47] == 'Layer 2 Down':
            input[i,47] = 4
        elif input[i,47] == 'Layer 3 Up':
            input[i,47] = 5
        elif input[i,47] == 'Layer 3 Down':
            input[i,47] = 6
        elif input[i, 47] == 'Repositioning':
            input[i,47] = 7
        elif input[i, 47] == 'End' or input[i, 47] == 'end':
            input[i,47] = 8
        elif input[i,47] == 'Starting':
            input[i,47] = 9
    return input

# train 데이터 전처리
def modify_train_data(input):
    # Modigying train.cs for train
    # - [tool_condition] : unworn/worn -> 0 / 1
    # - [item_inspection] : machining_finalize & passed -> yes & yes / yes & no / no : 0 / 1 / 2
    # - delete 'material' column and 'No' column
    train_sample_info = np.array(input.copy())
    train_sample_info = tool_condition(train_sample_info)
    train_sample_info = item_inspection(train_sample_info)
    # print(train_sample_info)

    train_sample_info = np.delete(train_sample_info,5,1)
    train_sample_info = np.delete(train_sample_info,0,1)
    train_sample_info = np.delete(train_sample_info,0,1)
    # print(train_sample_info)

    return train_sample_info

# 테스트 데이터 전처리
def read_all_data(train_data, all_files):
    """Reads all CSV files from the specified directory and categorizes them based on the train data results."""
    k = 0
    li_pass = []
    li_pass_half = []
    li_fail = []
    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, encoding="utf-8")
        
        # train_data의 k번째 행의 마지막 열이 결과
        result = train_data[k][-1]
        
        if result == 0:  # 합격
            li_pass.append(df)
        elif result == 1:  # 불합격
            li_pass_half.append(df)
        elif result == 2:  # 미완료
            li_fail.append(df)
        k += 1

    if li_pass:
        frame01 = pd.concat(li_pass, axis=0, ignore_index=True)
        data_pass = frame01.to_numpy()
        print("공정완료 및 육안검사 합격한 전체 데이터 수:", len(data_pass))
    else:
        data_pass = np.array([])
        print("합격 데이터 없음")

    if li_pass_half:
        frame02 = pd.concat(li_pass_half, axis=0, ignore_index=True)
        data_pass_half = frame02.to_numpy()
        print("공정완료 및 육안검사 불합격한 전체 데이터 수:", len(data_pass_half))
    else:
        data_pass_half = np.array([])
        print("불합격 데이터 없음")

    if li_fail:
        frame03 = pd.concat(li_fail, axis=0, ignore_index=True)
        data_fail = frame03.to_numpy()
        print("공정 미완료한 전체 데이터 수:", len(data_fail))
    else:
        data_fail = np.array([])
        print("미완료 데이터 없음")

    # print("\n\n")
    # print(data_pass.shape)
    # print(data_pass_half.shape)
    # print(data_fail.shape)

    return data_pass, data_pass_half, data_fail

def dataset(data_pass, data_pass_half, data_fail):
    # 데이터셋 구성
    data01 = data_pass[0:3228+6175,:] # 양품
    data02 = data_pass_half[0:6175,:] # 불량품
    data03 = data_fail[0:3228,:] # 불량품
    data = np.concatenate((data01,data02),axis=0)
    data = np.concatenate((data,data03),axis=0)
    data_all = data_pass[3228+6175:22645,:] # 평가 데이터

    return data, data_all

def scaler(data, data_all):
    sc = MinMaxScaler() # 각 열의 데이터에 대한 최소값과 최대값을 수하고 그 열의 데이터 값 각각을 0~1 사이의 값으로 조정
    X_train = sc.fit_transform(data)
    X_train = np.array(X_train)
    X_test = sc.fit_transform(data_all)
    X_test = np.array(X_test)

    return X_train, X_test

def label_data(X_train, X_test):
    Y_train = np.zeros((len(X_train),1),dtype='int') # 절반은 양품 절반은 불량
    Y_test = np.zeros((len(X_test),1),dtype='int') # 전부 양품
    I = int(Y_train.shape[0]/2)
    Y_train[0:I,:] = 0
    Y_train[I:I*2,:] = 1

    return Y_train, Y_test

def main():
    # 데이터 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)
    
    # train.csv 로드
    train_sample = pd.read_csv(os.path.join(base_path, "train.csv"), header=0, encoding="utf-8")
    train_sample_np = np.array(train_sample)
    
    # 실험 데이터 파일 목록 가져오기
    path = os.path.join(base_path, 'CNC Virtual Data set _v2')
    all_files = sorted(glob.glob(path + "/*.csv"))  # 파일 정렬하여 순서 보장
    
    # 데이터 개수 확인
    nb_pass, nb_pass_half, nb_defective = count_pass_fail(train_sample_np)
    print(f"\n합격: {nb_pass}, 불합격: {nb_pass_half}, 미완료: {nb_defective}")
    
    # train 데이터 전처리
    train_sample_info = modify_train_data(train_sample_np)
    
    # 테스트 데이터 분류
    data_pass, data_pass_half, data_fail = read_all_data(train_sample_info, all_files)

    # 공정 숫자형식으로 변경
    data_pass = machining_process(data_pass)
    data_pass_half = machining_process(data_pass_half)
    data_fail = machining_process(data_fail)

    # 데이터 셋 구성
    data, data_all = dataset(data_pass, data_pass_half, data_fail)

    # 데이터정제
    X_train, X_test = scaler(data, data_all)

if __name__ == "__main__":
    main()