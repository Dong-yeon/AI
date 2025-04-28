from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from tabulate import tabulate

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")

def plot_monthly_qty_by_customer(df):
    """월별 고객사별 수주 수량을 시각화하는 함수

    Args:
        df (pandas.DataFrame): 수주 데이터가 포함된 데이터프레임

    Returns:
        None: matplotlib 그래프를 출력합니다.
    """

    def prepare_data(df):
        """데이터 전처리를 수행하는 내부 함수"""
        df = df.copy()
        df['PLAN_DT'] = pd.to_datetime(df['PLAN_DT'])
        df['YEAR_MONTH'] = df['PLAN_DT'].dt.to_period('M').astype(str)
        return df.groupby(['YEAR_MONTH', 'CUST_NM'])['QTY'].sum().reset_index()

    def setup_plot():
        """그래프 기본 설정을 수행하는 내부 함수"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.grid(True)
        ax.set_title("월별 고객사 수주 수량")
        ax.set_xlabel("수주 월")
        ax.set_ylabel("수량")
        return fig, ax

    def plot_customer_data(grouped_data, ax):
        """고객사별 데이터를 그래프에 그리는 내부 함수"""
        for cust_nm in grouped_data['CUST_NM'].unique():
            cust_data = grouped_data[grouped_data['CUST_NM'] == cust_nm]
            ax.plot(cust_data['YEAR_MONTH'],
                    cust_data['QTY'],
                    marker='o',
                    label=cust_nm)

    # 메인 로직
    grouped_data = prepare_data(df)
    # fig, ax = setup_plot()
    # plot_customer_data(grouped_data, ax)
    #
    # plt.xticks(rotation=45)
    # plt.legend(title="고객사")
    # plt.tight_layout()
    # plt.show()

    # seaborn을 사용한 시각화
    plt.figure(figsize=(12, 6))

    # seaborn의 lineplot을 사용하여 고객사별 수량을 시각화
    sns.lineplot(data=grouped_data,
                    x='YEAR_MONTH',
                    y='QTY',
                    hue='CUST_NM',
                    marker='o')

    # 그래프 제목 및 레이블 설정
    plt.title("월별 고객사 수주 수량", fontsize=16)
    plt.xlabel("수주 월", fontsize=12)
    plt.ylabel("수량", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="고객사", fontsize=10)
    plt.tight_layout()
    plt.show()


def forecast_monthly_qty(df, target_month='2025-04'):
    """고객사별 월간 수량을 예측하는 함수

    Args:
        df (pandas.DataFrame): 예측에 사용할 데이터프레임
        target_month (str): 예측 대상 월 (기본값: '2025-04')

    Returns:
        pandas.DataFrame: 고객사별 예측 결과
    """

    def prepare_data(df):
        """데이터 전처리를 수행하는 함수"""
        df = df.copy()
        df['PLAN_DT'] = pd.to_datetime(df['PLAN_DT'])
        df['YEAR_MONTH'] = df['PLAN_DT'].dt.to_period('M').dt.to_timestamp()
        return df

    def create_monthly_data(cust_df):
        """고객사별 월간 데이터를 생성하는 함수"""
        monthly = cust_df.groupby('YEAR_MONTH')['QTY'].sum().reset_index()
        monthly.columns = ['ds', 'y']  # Prophet 형식으로 변환
        return monthly

    def predict_customer_qty(monthly_data, cust_nm):
        """개별 고객사의 수량을 예측하는 함수"""
        if len(monthly_data) < 2:
            print(f"[경고] {cust_nm} 고객사의 데이터가 2건 미만입니다.")
            return None

        try:
            model = Prophet()
            model.fit(monthly_data)
            future = model.make_future_dataframe(periods=1, freq='MS')
            forecast = model.predict(future)

            forecast['year_month'] = forecast['ds'].dt.strftime('%Y-%m')
            pred_row = forecast[forecast['year_month'] == target_month]

            if not pred_row.empty and not pd.isna(pred_row['yhat'].values[0]):
                pred_qty = pred_row['yhat'].values[0]
                print(f"[예측] {cust_nm} 고객사의 {target_month} 예측 수량: {round(pred_qty, 2)}")
                return round(pred_qty, 2)

            print(f"[경고] {cust_nm} 고객사의 예측값이 없습니다.")
            return 0

        except Exception as e:
            print(f"[오류] {cust_nm} 고객사 예측 중 오류 발생: {str(e)}")
            return None

    # 메인 로직
    processed_df = prepare_data(df)
    forecast_results = []

    for cust_nm in processed_df['CUST_NM'].unique():
        cust_df = processed_df[processed_df['CUST_NM'] == cust_nm]
        monthly_data = create_monthly_data(cust_df)

        predicted_qty = predict_customer_qty(monthly_data, cust_nm)
        forecast_results.append({
            'CUST_NM': cust_nm,
            'PREDICTED_QTY_2025_04': predicted_qty
        })

    return pd.DataFrame(forecast_results)


def analyze_process_times(df1, df2, export_excel=False):
    """작업 및 공정 시간을 분석하는 함수

    Args:
        df1 (pandas.DataFrame): 작업 주문 데이터
        df2 (pandas.DataFrame): 공정 데이터
        export_excel (bool): 엑셀 파일 저장 여부

    Returns:
        pandas.DataFrame: 분석 결과
    """
    def prepare_datetime_columns(df):
        """날짜/시간 컬럼을 전처리하는 함수"""
        df = df.copy()
        for col in ['WORK_START_DT', 'WORK_END_DT']:
            df[col] = pd.to_datetime(df[col])
        return df

    def calculate_process_times(df):
        """공정별 시간을 계산하는 함수"""
        df['계획종료시간'] = df['WORK_START_DT'] + pd.to_timedelta(df['TIME'], unit='m')
        overtime = (df['WORK_END_DT'] - df['계획종료시간']).dt.total_seconds() / 60
        df['오버된시간(분)'] = overtime.apply(lambda x: max(0, x))
        return df

    def merge_and_select_columns(proc_df, order_df):
        """데이터프레임 병합 및 컬럼 선택"""
        merged_df = pd.merge(
            proc_df,
            order_df,
            on='WORK_ORDER_MGT_SEQ',
            suffixes=('_공정', '_작업')
        )

        selected_columns = [
            'WORK_ORDER_MGT_SEQ', 'PART_NO', 'BOP_CD', 'BOP_NO',
            'WORK_START_DT_공정', 'WORK_END_DT_공정', 'TIME',
            '계획종료시간', '오버된시간(분)', 'WORK_START_DT_작업', 'WORK_END_DT_작업'
        ]

        new_column_names = [
            '작업번호', '품번', '공정코드', '공정번호',
            '공정시작시간', '공정종료시간', '예상소요시간(분)',
            '예상종료시간', '오버된시간(분)', '작업시작시간', '작업종료시간'
        ]

        result = merged_df[selected_columns].copy()
        result.columns = new_column_names
        return result

    def export_results(df):
        """분석 결과를 출력하고 필요시 엑셀로 저장"""
        print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

        if export_excel:
            df.to_excel('공정오버시간분석.xlsx', index=False)

    # 메인 로직
    try:
        # 1. 데이터 전처리
        order_df = prepare_datetime_columns(df1)
        proc_df = prepare_datetime_columns(df2)

        # 2. 공정 시간 계산
        proc_df = calculate_process_times(proc_df)

        # 3. 데이터 병합 및 결과 생성
        result_df = merge_and_select_columns(proc_df, order_df)

        # 4. 결과 출력 및 저장
        export_results(result_df)

        return result_df

    except Exception as e:
        print(f"[오류] 시간 분석 중 오류 발생: {str(e)}")
        return None


def estimate_process_times(learn_df, target_df):
    """공정별 예상 소요 시간을 계산하고 예측하는 함수

    Args:
        learn_df (pandas.DataFrame): 학습용 공정 데이터
        target_df (pandas.DataFrame): 예측 대상 공정 데이터

    Returns:
        pandas.DataFrame: 공정별 예상 시간 예측 결과
    """
    def prepare_learning_data(df):
        """학습 데이터 전처리"""
        df = df.copy()
        for col in ['WORK_START_DT', 'WORK_END_DT']:
            df[col] = pd.to_datetime(df[col])

        df['DURATION_MIN'] = (
                                     df['WORK_END_DT'] - df['WORK_START_DT']
                             ).dt.total_seconds() / 60

        return df

    def calculate_average_times(df):
        """공정별 평균 소요시간 계산"""
        return (df.groupby(['BOP_CD', 'BOP_NO'])['DURATION_MIN']
                .mean()
                .reset_index()
                .rename(columns={'DURATION_MIN': 'AVG_TIME'}))

    def predict_process_times(target_df, avg_times):
        """공정별 예상 시작/종료 시간 예측"""
        results = []
        current_time = pd.Timestamp.now()
        last_seq = None

        for _, row in target_df.iterrows():
            seq = row['WORK_ORDER_MGT_SEQ']
            if seq != last_seq:
                current_time = pd.Timestamp.now()
                last_seq = seq

            duration = row['AVG_TIME'] if not pd.isna(row['AVG_TIME']) else 0
            end_time = current_time + timedelta(minutes=duration)

            results.append({
                'WORK_ORDER_MGT_SEQ': seq,
                'ITEM_CD': row['ITEM_CD'],
                'BOP_CD': row['BOP_CD'],
                'BOP_NO': row['BOP_NO'],
                '예상시작시간': current_time,
                '예상종료시간': end_time,
                '예상소요시간(분)': duration
            })

            current_time = end_time

        return pd.DataFrame(results)

    def display_results(df):
        """결과 출력"""
        print("\n📋 공정 예상 시간 결과:")
        print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

    try:
        # 1. 학습 데이터 전처리
        processed_learn_df = prepare_learning_data(learn_df)

        # 2. 평균 소요시간 계산
        avg_time_map = calculate_average_times(processed_learn_df)

        # 3. 대상 데이터 준비
        target_with_avg = pd.merge(
            target_df.sort_values(['WORK_ORDER_MGT_SEQ', 'MIN_SORT']),
            avg_time_map,
            on=['BOP_CD', 'BOP_NO'],
            how='left'
        )

        # 4. 시간 예측
        result_df = predict_process_times(target_with_avg, avg_time_map)

        # 5. 결과 출력
        display_results(result_df)

        return result_df

    except Exception as e:
        print(f"[오류] 공정 시간 예측 중 오류 발생: {str(e)}")
        return None


def 불량률예측(df):
    recommended = df.sort_values("DEFECT_RATE", ascending=True).reset_index(drop=True)

    print("\n📋 불량률 결과:")
    print(tabulate(recommended, headers='keys', tablefmt='pretty', showindex=False))
